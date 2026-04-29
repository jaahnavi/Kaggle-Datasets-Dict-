
"""
RANKING STEP — run after kaggle_search_updated.py produces filtered_datasets.json.

Reads  : filtered_datasets.json
Outputs: dataset_lookup.json

Each entry in the output has:
    title, url, description, votes, downloads, usability, final_score, tags

tags consolidates: ai_technique, ai_domain, Kaggle dataset tags,
                   skill names, and role IDs — for skill-gap lookup.

Usage:
    python rank_datasets.py
    python rank_datasets.py --repo filtered_datasets.json --output dataset_lookup.json
"""

import json
import argparse
from pathlib import Path

import numpy as np
import xgboost as xgb
import joblib

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEFAULT_REPO_FILE   = Path("filtered_datasets.json")
DEFAULT_OUTPUT_FILE = Path("dataset_lookup.json")
MODEL_FILE          = "xgb_ranker.json"
SCALER_FILE         = "scaler.pkl"

PRIORITY_WEIGHT = {"critical": 3, "high": 2, "medium": 1}


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

def load_repo(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"❌  '{path}' not found. Run kaggle_search_updated.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        repo = json.load(f)
    print(f"📂  Loaded repo: {repo['total_unique_datasets']} unique datasets "
          f"across {len(repo['roles_included'])} roles\n")
    return repo


def load_model(model_file: str, scaler_file: str):
    model = xgb.Booster()
    model.load_model(model_file)
    scaler = joblib.load(scaler_file)
    print(f"✅  Model loaded: {model_file}")
    print(f"✅  Scaler loaded: {scaler_file}\n")
    return model, scaler


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def get_priority_score(dataset: dict) -> int:
    """
    Derive the highest priority score from all roles that reference this dataset.
    A dataset referenced by a 'critical' skill outranks one only referenced by 'medium'.
    """
    score = 1  # default: medium
    for ref in dataset.get("referenced_by", []):
        p = PRIORITY_WEIGHT.get(ref.get("category_priority", "medium"), 1)
        if p > score:
            score = p
    return score


def build_feature_matrix(repo: dict):
    """
    Build feature matrix X and parallel metadata list.

    Returns:
        X            — np.ndarray shape (N, 4)
        dataset_urls — list of N URLs (index aligns with X rows)
    """
    X, dataset_urls = [], []

    for url, ds in repo["datasets"].items():
        usability = ds.get("usability")
        if usability is None:
            continue  # skip datasets with no usability score

        priority_score = get_priority_score(ds)

        X.append([
            np.log1p(ds.get("votes",     0) or 0),
            np.log1p(ds.get("downloads", 0) or 0),
            float(usability),
            #float(priority_score),
        ])
        dataset_urls.append(url)

    X = np.array(X, dtype=np.float32)
    print(f"📊  Feature matrix: {X.shape}  "
          f"({len(repo['datasets']) - len(dataset_urls)} datasets skipped — no usability score)\n")
    return X, dataset_urls


# ─────────────────────────────────────────────
# SCORE + ATTACH
# ─────────────────────────────────────────────

def score_datasets(model, scaler, X: np.ndarray, dataset_urls: list, repo: dict) -> dict:
    """
    Run model inference and attach scores + cross_role_boost to each dataset.
    Returns the repo datasets dict with scores attached.
    """
    X_scaled = scaler.transform(X)
    raw_scores = model.predict(xgb.DMatrix(X_scaled))

    scored = {}
    for i, url in enumerate(dataset_urls):
        ds = repo["datasets"][url].copy()
        cross_role   = ds.get("cross_role_count", 1)
        model_score  = float(raw_scores[i])

        # Boost by cross_role_count: datasets useful across more roles rank higher
        # Small additive boost (0.05 per additional role beyond 1) so it doesn't
        # overpower the model score but acts as a meaningful tie-breaker
        #final_score  = model_score + 0.05 * (cross_role - 1)

        priority_score = get_priority_score(ds)
        final_score = model_score + 0.05 * (cross_role - 1) + 0.1 * (priority_score - 1)


        ds["model_score"]      = round(model_score, 6)
        ds["cross_role_boost"] = round(0.05 * (cross_role - 1), 6)
        ds["final_score"]      = round(final_score, 6)
        scored[url] = ds

    return scored


# ─────────────────────────────────────────────
# BUILD LOOKUP
# ─────────────────────────────────────────────

def build_lookup(scored: dict) -> list:
    """
    Flatten scored datasets into a list sorted by final_score.
    All metadata useful for skill-gap lookup is merged into a single tags list:
        ai_technique, ai_domain, Kaggle dataset tags, skill names, role IDs.
    """
    lookup = []
    for url, ds in scored.items():
        refs = ds.get("referenced_by", [])

        tags: set[str] = set()
        technique = ds.get("ai_technique", "")
        domain    = ds.get("ai_domain", "")
        if technique:
            tags.add(technique)
        if domain:
            tags.add(domain)
        tags.update(t for t in ds.get("tags", []) if t)
        tags.update(r["skill"]   for r in refs)
        tags.update(r["role_id"] for r in refs)

        lookup.append({
            "title":       ds["title"],
            "url":         url,
            "description": ds.get("description", ""),
            "votes":       ds.get("votes"),
            "downloads":   ds.get("downloads"),
            "usability":   ds.get("usability"),
            "final_score": ds.get("final_score"),
            "tags":        sorted(tags),
        })

    lookup.sort(key=lambda x: x["final_score"] or 0, reverse=True)
    return lookup


# ─────────────────────────────────────────────
# SAVE + SUMMARY
# ─────────────────────────────────────────────

def save_lookup(lookup: list, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, ensure_ascii=False)
    print(f"\n💾  Saved: {output_path}  ({len(lookup)} datasets)")


def print_summary(lookup: list) -> None:
    print(f"\n{'='*60}")
    print(f"🏆  Top 10 datasets by final score")
    print(f"{'='*60}")
    for i, entry in enumerate(lookup[:10], 1):
        print(f"  {i}. {entry['title'][:60]}")
        print(f"     score={entry['final_score']:.4f}  "
              f"votes={entry['votes']}  usability={entry['usability']}")
    print(f"\n👉  {DEFAULT_OUTPUT_FILE} is ready for the recommendation engine.\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rank filtered_datasets.json using saved XGBoost model → dataset_lookup.json"
    )
    parser.add_argument("--repo",   type=Path, default=DEFAULT_REPO_FILE,
                        help="Input filtered datasets (default: filtered_datasets.json)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE,
                        help="Output lookup file (default: dataset_lookup.json)")
    parser.add_argument("--model",  type=str,  default=MODEL_FILE,
                        help="XGBoost model file (default: xgb_ranker.json)")
    parser.add_argument("--scaler", type=str,  default=SCALER_FILE,
                        help="Scaler pickle file (default: scaler.pkl)")
    args = parser.parse_args()

    repo           = load_repo(args.repo)
    model, scaler  = load_model(args.model, args.scaler)

    X, dataset_urls = build_feature_matrix(repo)

    print("🤖  Scoring datasets with XGBoost ranker...")
    scored = score_datasets(model, scaler, X, dataset_urls, repo)

    print("📐  Building lookup...")
    lookup = build_lookup(scored)

    save_lookup(lookup, args.output)
    print_summary(lookup)


if __name__ == "__main__":
    main()
