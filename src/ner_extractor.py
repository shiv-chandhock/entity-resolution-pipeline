from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Label mapping matching the CoNLL-2003 training scheme
LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

MODEL_PATH = "./models/ner_filter/checkpoint-1756"

print("Loading NER model from trained checkpoint...")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
_model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
_model.eval()
print("NER model loaded successfully.")


def extract_entities(text: str) -> list[dict]:
    """
    Run NER inference on raw text and return detected entities.
    
    Uses the same sub-token merging strategy as app.py:
    - ## prefixed tokens are concatenated to the previous word
    - B-/I- tag sequences are grouped into entity spans
    
    Returns a list of dicts:
        [{"phrase": "Barack Obama", "entity_type": "PER"}, ...]
    """
    if not text or not text.strip():
        return []

    # Tokenize the input text
    inputs = _tokenizer(text, return_tensors="pt", truncation=True)

    # Run inference (no gradient computation needed)
    with torch.no_grad():
        outputs = _model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    input_ids = inputs["input_ids"].squeeze().tolist()
    tokens = _tokenizer.convert_ids_to_tokens(input_ids)

    # Handle edge case: single-token input
    if isinstance(predictions, int):
        predictions = [predictions]
        tokens = [tokens]

    # Merge sub-tokens and group B-/I- tag sequences into entities
    extracted_entities = []
    current_entity = []
    current_label = None

    for token, pred_idx in zip(tokens, predictions):
        label = LABEL_LIST[pred_idx]

        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]", "CLS", "SEP", "PAD"]:
            continue

        # Handle ## sub-word tokens — append to previous word
        if token.startswith("##"):
            if current_entity:
                current_entity[-1] = current_entity[-1] + token[2:]
            continue

        # B-* tag: start a new entity (flush any previous one first)
        if label.startswith("B-"):
            if current_entity:
                extracted_entities.append({
                    "phrase": " ".join(current_entity),
                    "entity_type": current_label,
                })
            current_entity = [token]
            current_label = label[2:]  # "B-PER" → "PER"

        # I-* tag: continue current entity only if types match
        elif label.startswith("I-") and current_label == label[2:]:
            current_entity.append(token)

        # O tag or type mismatch: flush current entity
        else:
            if current_entity:
                extracted_entities.append({
                    "phrase": " ".join(current_entity),
                    "entity_type": current_label,
                })
                current_entity = []
                current_label = None

    # Don't forget the last entity
    if current_entity:
        extracted_entities.append({
            "phrase": " ".join(current_entity),
            "entity_type": current_label,
        })

    # Deduplicate: keep only unique (phrase, type) pairs, preserving order
    seen = set()
    unique_entities = []
    for entity in extracted_entities:
        key = (entity["phrase"].lower(), entity["entity_type"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)

    return unique_entities
