# Hybrid Contextual Entity Resolver (v2.0)

A production-grade, NER-powered entity resolution pipeline that detects named entities in unstructured text using a fine-tuned **DistilBERT** model, then resolves them to precise **Wikidata Q-IDs** through a multi-signal disambiguation engine.

---

## 🧠 How It Works

```
Raw Text → DistilBERT NER → Entity Detection → Wikidata API → Multi-Signal Resolver → Resolved Entities
```

The pipeline operates across three intelligent stages:

### Stage 1 — NER Entity Detection
A **DistilBERT** model fine-tuned on the CoNLL-2003 dataset (**91.56% F1 score**) identifies named entities and classifies them into four categories:

| Tag | Description | Example |
|-----|-------------|---------|
| `PER` | Person | Barack Obama, Angela Merkel |
| `ORG` | Organization | SpaceX, Apple, United Nations |
| `LOC` | Location | Berlin, California, Amazon |
| `MISC` | Miscellaneous | iPhone, Grammy Awards, English |

### Stage 2 — Wikidata Candidate Retrieval
Each detected entity is queried against the **live Wikidata API** (`wbsearchentities`), retrieving up to 5 candidate matches with labels, descriptions, and Q-IDs.

### Stage 3 — Multi-Signal Disambiguation
When multiple Wikidata candidates exist, a **4-signal scoring system** resolves the correct one:

1. **Wikidata Rank Bias** — First results are the most prominent entities (e.g., "Barack Obama" the president ranks above "Barack Obama Sr.")
2. **Exact Name Match** — Candidates whose full name appears in the input context get boosted
3. **Stopword-Filtered Jaccard** — Word overlap between surrounding text and candidate descriptions, with 100+ stopwords filtered out for cleaner matching
4. **Description Keyword Matching** — Context words found in the Wikidata description contribute additional signal

---

## 📁 Project Structure

```
entity-resolver-pipeline/
├── app.py                          # FastAPI server (POST /extract endpoint)
├── models/
│   └── ner_filter/
│       ├── checkpoint-1756/        # Best trained model weights (epoch 2, 91.56% F1)
│       └── checkpoint-878/         # Epoch 1 checkpoint
├── src/
│   ├── ner_extractor.py            # Loads DistilBERT, runs NER inference, merges sub-tokens
│   ├── pipeline.py                 # Orchestrates NER → Wikidata → Resolver flow
│   ├── api_client.py               # Wikidata API client (wbsearchentities)
│   ├── detector.py                 # Ambiguity detection (unique vs. multiple matches)
│   ├── resolver.py                 # Multi-signal disambiguation engine
│   ├── stream_buffer.py            # Text tokenization utilities
│   └── train_ner.py                # NER model training script (DistilBERT + CoNLL-2003)
├── config/
│   └── __init__.py
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Active network connection (for live Wikidata queries)
- ~500MB disk space (for model weights)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd entity-resolver-pipeline

# 2. Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install torch transformers datasets evaluate seqeval fastapi uvicorn numpy
```

### Run the API Server

```bash
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

The server starts at `http://127.0.0.1:8000`. Interactive API docs are available at `http://127.0.0.1:8000/docs`.

### Make a Request

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Elon Musk founded SpaceX in Los Angeles"}'
```

### Example Response

```json
{
  "status": "success",
  "input_processed": "Elon Musk founded SpaceX in Los Angeles",
  "total_extracted": 3,
  "entities": [
    {
      "extracted_text": "Elon Musk",
      "entity_type": "PER",
      "wikidata_id": "Q317521",
      "canonical_name": "Elon Musk",
      "resolution_status": "disambiguated_via_jaccard",
      "jaccard_confidence": 0.65
    },
    {
      "extracted_text": "SpaceX",
      "entity_type": "ORG",
      "wikidata_id": "Q193701",
      "canonical_name": "SpaceX",
      "resolution_status": "disambiguated_via_jaccard",
      "jaccard_confidence": 0.65
    },
    {
      "extracted_text": "Los Angeles",
      "entity_type": "LOC",
      "wikidata_id": "Q65",
      "canonical_name": "Los Angeles",
      "resolution_status": "disambiguated_via_jaccard",
      "jaccard_confidence": 0.827
    }
  ]
}
```

---

## 🏋️ Training the NER Model

The model is already trained and saved at `models/ner_filter/checkpoint-1756`. To retrain from scratch:

```bash
python src/train_ner.py
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `distilbert-base-cased` |
| Dataset | CoNLL-2003 (14,041 train / 3,250 val / 3,453 test) |
| Epochs | 2 (with early stopping, patience=2) |
| Learning Rate | 2e-5 (linear decay, 10% warmup) |
| Batch Size | 16 |
| Best Model Selection | F1 score |

### Training Results

| Metric | Score |
|--------|-------|
| **Overall F1** | **91.56%** |
| Precision | 90.54% |
| Recall | 92.61% |
| PER F1 | 95.85% |
| ORG F1 | 87.86% |
| LOC F1 | 94.33% |
| MISC F1 | 82.86% |

---

## 🔧 API Reference

### `POST /extract`

Extract and resolve named entities from text.

**Request Body:**
```json
{
  "text": "string (required)"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `extracted_text` | string | The entity phrase detected by NER |
| `entity_type` | string | NER classification: `PER`, `ORG`, `LOC`, or `MISC` |
| `wikidata_id` | string | Resolved Wikidata Q-ID (e.g., `Q317521`) |
| `canonical_name` | string | Official Wikidata label |
| `resolution_status` | string | `resolved_unique`, `disambiguated_via_jaccard`, or `no_wikidata_match` |
| `jaccard_confidence` | float | Disambiguation confidence score (0.0 – 1.0) |

---

## 📝 Version History

| Version | Changes |
|---------|---------|
| **v2.0** | DistilBERT NER model integration, multi-signal resolver, FastAPI endpoint |
| v1.0 | N-gram brute-force candidate generation, basic Jaccard resolver |

---



This project is for educational and experimental purposes.
