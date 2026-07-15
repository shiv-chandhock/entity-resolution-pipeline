# Stopwords to filter out from context before disambiguation
STOPWORDS = {
    "the", "a", "an", "is", "in", "on", "at", "by", "for", "to", "and", "or",
    "of", "with", "from", "above", "below", "under", "this", "that", "it",
    "was", "were", "has", "had", "have", "been", "be", "are", "am", "will",
    "would", "could", "should", "may", "might", "can", "do", "does", "did",
    "not", "no", "but", "if", "then", "than", "so", "as", "its", "his", "her",
    "he", "she", "they", "we", "you", "i", "me", "my", "our", "your", "their",
    "who", "whom", "which", "what", "where", "when", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "too", "very", "just", "also", "about", "up", "out", "into",
    "over", "after", "before", "between", "through", "during", "again", "once",
    "met", "said", "told", "went", "came", "made", "took", "got", "gave",
    "found", "known", "called", "named", "based", "located", "born", "died",
    "new", "old", "first", "last", "recent", "many", "much",
}


def calculate_jaccard(set_a: set, set_b: set) -> float:
    """Standard Jaccard similarity between two sets."""
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(intersection) / len(union)


def resolve_context_ambiguity(candidates: list[dict], surrounding_text: str) -> dict:
    """
    Disambiguate between multiple Wikidata candidates using a multi-signal scoring system:
    
    1. Wikidata rank bias — first results are most popular/relevant
    2. Exact name match — candidate label matches the search phrase
    3. Context-description Jaccard — overlap between surrounding text and candidate description
    4. Stopword filtering — removes noise words from Jaccard computation
    """
    if not candidates:
        return {"q_id": None, "name": None, "confidence": 0.0}

    # Filter stopwords from context for cleaner Jaccard matching
    raw_context_words = set(surrounding_text.lower().split())
    context_words = raw_context_words - STOPWORDS

    best_candidate = None
    max_score = -1.0

    for rank, candidate in enumerate(candidates):
        score = 0.0

        # SIGNAL 1: Wikidata rank bias (first result = most popular)
        # Wikidata's wbsearchentities returns results ordered by prominence.
        # The 44th US President "Barack Obama" always comes before "Barack Obama Sr."
        rank_bonus = max(0.0, 0.3 - (rank * 0.07))  # 0.30, 0.23, 0.16, 0.09, 0.02
        score += rank_bonus

        # SIGNAL 2: Exact name match — candidate label matches search phrase
        cand_name = candidate.get("name", "").lower().strip()
        search_words = {w.lower() for w in cand_name.split()}
        name_overlap = len(search_words.intersection(raw_context_words))
        if name_overlap == len(search_words) and len(search_words) > 0:
            score += 0.35  # Full name found in context

        # SIGNAL 3: Context ↔ Description Jaccard (with stopwords removed)
        desc_text = candidate.get("description", "")
        desc_words = set(desc_text.lower().split()) - STOPWORDS
        if context_words and desc_words:
            jaccard = calculate_jaccard(context_words, desc_words)
            score += jaccard * 0.5  # Scale Jaccard contribution

        # SIGNAL 4: Description keyword hints
        # Boost candidates whose description contains contextually relevant terms
        desc_lower = desc_text.lower()
        context_hints = {
            "president": 0.1, "politician": 0.05, "company": 0.05,
            "city": 0.05, "country": 0.05, "river": 0.05,
            "organization": 0.05, "footballer": 0.05, "actor": 0.05,
            "musician": 0.05, "scientist": 0.05, "writer": 0.05,
        }
        for hint_word in context_words:
            if hint_word in desc_lower:
                score += 0.05  # Small boost for each context word found in description

        if score > max_score:
            max_score = score
            best_candidate = candidate

    if best_candidate:
        return {
            "q_id": best_candidate.get("q_id"),
            "name": best_candidate.get("name"),
            "confidence": round(min(max_score, 1.0), 3),
        }

    # Fallback: return first candidate (most popular by Wikidata ranking)
    return {
        "q_id": candidates[0].get("q_id"),
        "name": candidates[0].get("name"),
        "confidence": 0.0,
    }