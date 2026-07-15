import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.api_client import fetch_live_candidates
from src.detector import eval_phrase_ambiguity
from src.resolver import resolve_context_ambiguity

app = FastAPI(
    title="Hybrid Live Wikidata Entity-Resolution Pipeline",
    version="2.0"
)

MODEL_PATH = "./models/ner_filter/checkpoint-1756"
LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_PATH,
    num_labels=9,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)
model.eval()

class TextPayload(BaseModel):
    text: str

@app.post("/extract")
async def extract_and_resolve_entities(payload: TextPayload):
    raw_text = payload.text
    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=400, detail="Input text string cannot be empty.")

    inputs = tokenizer(raw_text, return_tensors="pt", truncation=True, is_split_into_words=False)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    input_ids = inputs["input_ids"].squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    if isinstance(predictions, int):
        predictions = [predictions]
        tokens = [tokens]

    extracted_entities = []
    current_entity = []
    current_label = None

    for token, pred_idx in zip(tokens, predictions):
        label = LABEL_LIST[pred_idx]
        
        if token in ["CLS", "SEP", "PAD", "[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        if token.startswith("##"):
            if current_entity:
                current_entity[-1] = current_entity[-1] + token[2:]
            continue

        if label.startswith("B-"):
            if current_entity:
                extracted_entities.append((" ".join(current_entity), current_label))
            current_entity = [token]
            current_label = label[2:]  
        elif label.startswith("I-") and current_label == label[2:]:
            current_entity.append(token)
        else:
            if current_entity:
                extracted_entities.append((" ".join(current_entity), current_label))
                current_entity = []
                current_label = None
                
    if current_entity:
        extracted_entities.append((" ".join(current_entity), current_label))

    final_resolved_payload = []

    for phrase, entity_type in extracted_entities:
        detection = eval_phrase_ambiguity(phrase)
        status = detection["status"]
        
        if status in ["no_match", "empty"]:
            final_resolved_payload.append({
                "extracted_text": phrase,
                "entity_type": entity_type,
                "wikidata_id": None,
                "canonical_name": None,
                "resolution_status": "no_wikidata_match",
                "jaccard_confidence": 0.0
            })
            continue

        if status == "resolved_unique":
            unique_cand = detection["candidates"][0]
            final_resolved_payload.append({
                "extracted_text": phrase,
                "entity_type": entity_type,
                "wikidata_id": unique_cand.get("q_id"),
                "canonical_name": unique_cand.get("name"),
                "resolution_status": "resolved_unique",
                "jaccard_confidence": 1.0
            })

        elif status == "requires_resolution":
            resolved_entity = resolve_context_ambiguity(detection["candidates"], raw_text)
            final_resolved_payload.append({
                "extracted_text": phrase,
                "entity_type": entity_type,
                "wikidata_id": resolved_entity.get("q_id"),
                "canonical_name": resolved_entity.get("name"),
                "resolution_status": "disambiguated_via_jaccard",
                "jaccard_confidence": resolved_entity.get("confidence", 0.0)
            })

    return {
        "status": "success",
        "input_processed": raw_text,
        "total_extracted": len(final_resolved_payload),
        "entities": final_resolved_payload
    }