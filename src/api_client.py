import urllib.request
import urllib.parse
import json

WIKIDATA_ENDPOINT = "https://www.wikidata.org/w/api.php" 
def fetch_live_candidates(search_query: str) -> list[dict]:
    if not search_query or not search_query.strip():
        return []
    
    params = {
        "action":"wbsearchentities",
        "format":"json",
        "language":"en",
        "type":"item",
        "limit":5,
        "search": search_query.strip()
    }
    url_encoded_params = urllib.parse.urlencode(params)
    full_url = F"{WIKIDATA_ENDPOINT}?{url_encoded_params}"

    headers ={
        "User-Agent": "HybridEntityResolverExperiment/1.0 (Contact: shiv chandhock)"
    }

    try:
        request = urllib.request.Request(full_url, headers=headers)
        with urllib.request.urlopen(request, timeout=5) as response:
            raw_data = response.read().decode("utf-8")
            payload =  json.loads(raw_data)

            search_result = payload.get("search", [])

            processed_candidates = []
            for item in search_result:
                processed_candidates.append({
                    "q_id" : item.get("id"),
                    "name": item.get("label"),
                    "description": item.get("description","").lower()
                })

        return processed_candidates
    except Exception as e:
        print(f"\n[Network Warning] Live API fetch failed for '{search_query}': {e}")
        return[]
    