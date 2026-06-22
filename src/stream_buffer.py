def normalize_and_tokenize(text: str) -> list[str]:
    if not text:
        return []
    
    clean_text = "".join(char for char in text if char not in ".,!?()\"")
    lowercase_text = clean_text.lower()
    tokens = lowercase_text.split()
    return tokens

def generate_ngram_candidates(tokens: list[str], max_window: int = 3) -> list[dict]:
    candidates = []
    n = len(tokens)

    for i in range(n):
        for window_size in range(1, max_window + 1):
            if i + window_size <= n:
                phrase_slice = tokens[i : i + window_size]
                candidate_phrase = " ".join(phrase_slice)

                candidates.append({
                    "phrase": candidate_phrase,
                    "start_index": i,
                    "token_len": window_size
                })
    return candidates