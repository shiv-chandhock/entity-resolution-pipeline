import time
from src.ner_extractor import extract_entities
from src.detector import eval_phrase_ambiguity
from src.resolver import resolve_context_ambiguity


def run_hybrid_resolver_pipeline(raw_text: str) -> list[dict]:
    """
    NER-powered entity resolution pipeline.
    
    Flow: raw_text → DistilBERT NER → detected entities → Wikidata lookup → resolve ambiguity
    """
    if not raw_text or not raw_text.strip():
        return []

    # STAGE 1: NER Model extracts entity phrases and types
    ner_entities = extract_entities(raw_text)

    if not ner_entities:
        return []

    final_resolved_entities = []

    for ner_entity in ner_entities:
        phrase = ner_entity["phrase"]
        entity_type = ner_entity["entity_type"]

        # Skip very short or single-character detections (likely noise)
        if len(phrase.strip()) <= 1:
            continue

        # PACING BUFFER: Small delay between API calls to prevent rate limiting
        time.sleep(0.1)

        # STAGE 2: Wikidata API Lookup for each NER-detected entity
        detection = eval_phrase_ambiguity(phrase)
        status = detection["status"]

        if status in ["no_match", "empty"]:
            continue

        resolved_entity = None

        if status == "resolved_unique":
            unique_cand = detection["candidates"][0]
            resolved_entity = {
                "phrase": phrase,
                "entity_type": entity_type,
                "q_id": unique_cand.get("q_id"),
                "name": unique_cand.get("name"),
                "confidence": 1.0,
            }

        elif status == "requires_resolution":
            # STAGE 3: Jaccard-based disambiguation for ambiguous entities
            resolved_entity = resolve_context_ambiguity(detection["candidates"], raw_text)
            resolved_entity["phrase"] = phrase
            resolved_entity["entity_type"] = entity_type

        if resolved_entity and resolved_entity.get("q_id"):
            final_resolved_entities.append(resolved_entity)

    return final_resolved_entities