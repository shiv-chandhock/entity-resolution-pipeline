def calaculate_jaccard(set_a : set, set_b : set) -> float:
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(intersection)/len(union)

def resolve_context_ambiguity(candidates : list[dict], surrounding_text : str) -> dict:
    if not candidates:
        return {"q_id": None, "name": None, "confidence": 0.0}
    
    context_words = set(surrounding_text.lower().split())

    best_candidate = None
    max_score = -1.0

    for candidate in candidates:
        desc_text = candidate.get("description", "")
        desc_words = set(desc_text.split())

        score = calaculate_jaccard(context_words, desc_words)

        cand_name_lower = candidate.get("name", "").lower()
        if cand_name_lower in context_words:
            score += 0.1
        
        if score > max_score:
            max_score = score
            best_candidate = candidate

    if best_candidate and max_score > 0.0:
        return{
            "q_id": best_candidate.get("q_id"),
            "name": best_candidate.get("name"),
            "confidence": round(max_score, 3)
        }        
    return {
        "q_id": candidates[0].get("q_id"),
        "name": candidates[0].get("name"),
        "confidence": 0.0
    }