from src.api_client import fetch_live_candidates
def eval_phrase_ambiguity(phrase: str)-> dict:
    if not phrase or len(phrase.strip()) <= 1:
        return {"is_ambigous": False, "candidates":[], "status":"empty"}
    
    live_results = fetch_live_candidates(phrase)
    result_count = len(live_results)

    if result_count == 0:
        return {"is_ambiguous": False, "candidates": [], "status": "no_match"}
        
    elif result_count == 1:
        # Perfectly clean, unique match—no resolution logic needed
        return {"is_ambiguous": False, "candidates": live_results, "status": "resolved_unique"}
        
    else:
        # Multiple entities found! Flag for Jaccard evaluation
        return {"is_ambiguous": True, "candidates": live_results, "status": "requires_resolution"}