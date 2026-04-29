# Kaggle Dataset Ranker for Skill Gap Recommendation

Identifies the most relevant Kaggle datasets for a given skill gap in data roles.
Given a target role and skill set, the pipeline fetches datasets from Kaggle, filters them for relevance using an LLM, and ranks them using a pre-trained XGBoost model.
The output is a single JSON file tagged by skill, domain, technique, and role — ready to look up datasets for any specific skill gap.

Supported roles: Data Analyst, Business Analyst, Financial Analyst, Data Scientist, Data Engineer, BI Analyst, Marketing Analyst, Quantitative Analyst.

---

## How it works

1. `kaggle_search_updated.py` reads `role_skill_sets.json`, fans out Kaggle searches per skill using tag-based queries, filters results through Groq (Llama 3) for relevance and domain/technique classification, and consolidates everything into `filtered_datasets.json`.
2. `rank_datasets.py` reads `filtered_datasets.json`, scores each dataset using a saved XGBoost ranker (`xgb_ranker.json` + `scaler.pkl`), and outputs `dataset_lookup.json` — a flat list sorted by final score.

---

## Requirements

```
pip install kaggle xgboost scikit-learn numpy joblib requests
```

You also need:
- `~/.kaggle/kaggle.json` — Kaggle API credentials ([instructions](https://www.kaggle.com/docs/api))
- `GROQ_API_KEY` environment variable — free API key from [console.groq.com](https://console.groq.com)

---

## Usage

**Step 1 — Search and filter**

```bash
# All 8 roles
python kaggle_search_updated.py

# One or more specific roles
python kaggle_search_updated.py --role data_scientist
python kaggle_search_updated.py --role data_analyst financial_analyst
```

Output: `filtered_datasets.json`

**Step 2 — Rank**

```bash
python rank_datasets.py
```

Output: `dataset_lookup.json`

---

## Output format

`dataset_lookup.json` is a list of datasets sorted by `final_score` (highest first).

```json
{
  "title": "...",
  "url": "https://www.kaggle.com/datasets/...",
  "description": "...",
  "votes": 4114,
  "downloads": 509411,
  "usability": 1.0,
  "final_score": 2.52,
  "tags": ["SQL", "retail", "data_analyst", "nlp", "regression", ...]
}
```

`tags` consolidates the AI-assigned technique and domain, Kaggle's own dataset tags, the skill names that surfaced this dataset, and the roles it applies to. To find datasets for a specific skill gap, filter entries where `tags` contains the skill or role of interest.

---

## Files

| File | Description |
|---|---|
| `kaggle_search_updated.py` | Step 1: search, filter, consolidate |
| `rank_datasets.py` | Step 2: rank and produce lookup |
| `role_skill_sets.json` | Skills taxonomy for all 8 roles |
| `xgb_ranker.json` | Trained XGBoost ranking model |
| `scaler.pkl` | Feature scaler fitted during training |
