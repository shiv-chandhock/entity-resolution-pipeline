import time
from src.stream_buffer import normalize_and_tokenize, generate_ngram_candidates
from src.detector import eval_phrase_ambiguity
from src.resolver import resolve_context_ambiguity

# A collection of pure structural noise words to save network quota
STOP_WORDS = {
    "the", "a", "an", "is", "in", "on", "at", "by", "for", "to", "and", "or", 
    "of", "with", "from", "above", "below", "under", "this", "that", "it", "recent", "who"
}

def is_pure_noise(phrase: str) -> bool:
    """Returns True if the entire phrase is just structural grammar noise."""
    words = phrase.lower().split()
    return all(word in STOP_WORDS for word in words)

def run_hybrid_resolver_pipeline(raw_text : str) -> list[dict]:
    if not raw_text or not raw_text.strip():
        return []
    
    tokens = normalize_and_tokenize(raw_text)
    
    # MAX_WINDOW INCREASE: Bumped to 5 to safely capture longer proper titles
    raw_candidates = generate_ngram_candidates(tokens, max_window=5)
    raw_candidates.sort(key = lambda x: x["token_len"], reverse=True)

    final_resolved_entities = []
    processed_indices = set()

    for item in raw_candidates:
        phrase = item["phrase"]
        start_idx = item["start_index"]
        length = item["token_len"]

        # GATEKEEPER 1: Optimization to prevent HTTP 429 Rate Limiting
        if is_pure_noise(phrase):
            continue

        covered_indices = set(range(start_idx, start_idx+length))
        if not covered_indices.isdisjoint(processed_indices):
            continue

        # PACING BUFFER: Spacing out lookups to prevent aggressive IP blocking
        time.sleep(0.1)

        # GATEKEEPER 2: Live API Lookup
        detection = eval_phrase_ambiguity(phrase)
        status = detection["status"]
        if status in ["no_match", "empty"]:
            continue

        resolved_entity = None

        if status == "resolved_unique":
            unique_cand = detection["candidates"][0]
            resolved_entity = {
                "phrase" : phrase,
                "q_id": unique_cand.get("q_id"),
                "name": unique_cand.get("name"),
                "confidence": 1.0
            }

        elif status == "requires_resolution":
            resolved_entity = resolve_context_ambiguity(detection["candidates"], raw_text)
            resolved_entity["phrase"] = phrase

        if resolved_entity and resolved_entity.get("q_id"):
            final_resolved_entities.append(resolved_entity) 
            processed_indices.update(covered_indices)

    return final_resolved_entities