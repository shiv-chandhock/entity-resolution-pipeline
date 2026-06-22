# Hybrid Contextual Entity Resolver (v1.0)

An asynchronous, multi-layered pipeline designed to extract ambiguous text phrases, query them against the live Wikidata API, and resolve semantic context down to precise Wikidata Q-IDs.

## 🏗️ Architecture (V1)

The pipeline processes raw unstructured text across three decoupled layers:

1. **Layer 1 (Tokenization & Candidate Generation):** Cleans text inputs, strips trailing punctuation fragments, and generates multi-window sliding n-gram candidate arrays sorted by token length.
2. **Layer 2 (API Gatekeeper Filter):** Cross-references candidates against a localized `STOP_WORDS` noise gate to prevent unnecessary web traffic, applying a network pacing buffer to stay within endpoint rate-limiting parameters.
3. **Layer 3 (Context Math Engine):** Applies Jaccard Similarity algorithms to evaluate candidate description tokens against surrounding input text tokens, resolving real-world entity ambiguity.

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Active network connection for live Wikidata queries

### Installation & Run
1. Clone the repository and navigate to the project root.
2. Initialize and activate your virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate