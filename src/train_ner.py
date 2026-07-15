from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, EarlyStoppingCallback
import numpy as np
import evaluate

print("Fetching data from huggingface ...")
raw_datasets = load_dataset("lhoestq/conll2003")

print("Tokenizer loading ....")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != current_word:
            current_word = word_id
            new_labels.append(labels[word_id])
        else:
            # Propagate I-* labels to continuation sub-tokens
            # B-tags are odd indices (1,3,5,7) → convert to I-tag (B+1)
            label = labels[word_id]
            if label % 2 == 1:  # B-PER=1, B-ORG=3, B-LOC=5, B-MISC=7
                new_labels.append(label + 1)
            else:
                new_labels.append(label)
    return new_labels

def tokenize_and_align_batch(batch):
    tokenized_inputs = tokenizer(
        batch["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    aligned_labels = []
    for i, labels in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

print("\nMapping pipeline across all 14,042 rows...")
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_batch,
    batched=True,
    remove_columns=["id", "tokens", "pos_tags", "chunk_tags", "ner_tags"],
    load_from_cache_file=False
)

print("Tokenization dataset mapping completed successfully!")
print(tokenized_datasets)       

print("\nInitializing model architecture...")
num_labels = 9

try:
    print("Loading locally optimized model parameters from './models/ner_filter'...")
    model = AutoModelForTokenClassification.from_pretrained(
        "./models/ner_filter",
        num_labels=num_labels
    )
except Exception as e:
    print(f"Local model weights not found. Falling back to base architecture. Error: {e}")
    model = AutoModelForTokenClassification.from_pretrained(
        "distilbert-base-cased",
        num_labels=num_labels
    )

print("\nSetting up training arguments")
training_args = TrainingArguments(
    output_dir="./models/ner_filter",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
)

print("Initializing data collator... ")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    true_predictions = [
        [label_list[p_idx] for (p_idx, l_idx) in zip(prediction, label) if l_idx != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l_idx] for (p_idx, l_idx) in zip(prediction, label) if l_idx != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        "PER_f1": results.get("PER", {}).get("f1", 0.0),
        "ORG_f1": results.get("ORG", {}).get("f1", 0.0),
        "LOC_f1": results.get("LOC", {}).get("f1", 0.0),
        "MISC_f1": results.get("MISC", {}).get("f1", 0.0),
    }

print("Assembling trainer engine...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\n=======================================================")
print("TRAINING STARTED")
print("=======================================================")
trainer.train()
print("TRAINING COMPLETE")

print("\n=======================================================")
print("RUNNING TRACE SAMPLE CHECK (DIAGNOSTIC)")
print("=======================================================")

predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
preds_argmax = np.argmax(predictions, axis=-1)

for i in range(3):
    print(f"\n--- Sentence Sample {i+1} ---")
    raw_ids = tokenized_datasets["validation"][i]["input_ids"]
    text_tokens = tokenizer.convert_ids_to_tokens(raw_ids)
    
    true_tags = [label_list[l] if l != -100 else "PAD" for l in labels[i]]
    pred_tags = [label_list[p] if l != -100 else "PAD" for p, l in zip(preds_argmax[i], labels[i])]
    
    for tok, tru, prd in list(zip(text_tokens, true_tags, pred_tags))[:15]:
        if tru != "PAD":
            print(f"Token: {tok:<12} | True Tag (Evaluator POV): {tru:<8} | Predicted Tag (Model POV): {prd:<8}")

print("\n=======================================================")
print("RUNNING FINAL VALIDATION SCOREBOARD")
print("=======================================================")

eval_results = trainer.evaluate()

print("\n--- FINAL METRICS PERFORMANCE SCORES ---")
print(f"Global Token Accuracy : {eval_results['eval_accuracy'] * 100:.2f}%")
print(f"Overall Precision     : {eval_results['eval_precision'] * 100:.2f}%")
print(f"Overall Recall        : {eval_results['eval_recall'] * 100:.2f}%")
print(f"Overall F1-Score      : {eval_results['eval_f1'] * 100:.2f}%")
print("")
print("--- PER-ENTITY TYPE BREAKDOWN ---")
for etype in ["PER", "ORG", "LOC", "MISC"]:
    key = f"eval_{etype}_f1"
    score = eval_results.get(key, 0.0)
    print(f"  {etype:<6} F1 : {score * 100:.2f}%")
print("=======================================================")