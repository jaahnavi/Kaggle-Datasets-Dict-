"""
SEARCH + FILTER PIPELINE — run this once offline per role (or all roles at once).

What this script does:
  1. Reads role_skill_sets.json to get all skills + tags for each role
  2. Searches Kaggle using tag-based queries per skill
  3. AI-filters results via Groq/Llama3 for relevance + domain/technique tagging
  4. Consolidates all roles into a single filtered_datasets.json (deduplicated by URL)

Next step: run rank_datasets.py to score and produce dataset_lookup.json.

Requirements:
    pip install kaggle xgboost scikit-learn numpy joblib requests
    ~/.kaggle/kaggle.json       — Kaggle API credentials
    GROQ_API_KEY env var        — free at https://console.groq.com
    role_skill_sets.json        — the skills taxonomy file (same directory)

Usage:
    # Run all roles
    python kaggle_search_updated.py

    # Run a single role only
    python kaggle_search_updated.py --role data_scientist

    # Run multiple specific roles
    python kaggle_search_updated.py --role data_analyst financial_analyst
"""

import re
import json
import os
import time
import argparse
from pathlib import Path

import numpy as np
import requests
import kaggle

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GROQ_API_KEY             = os.getenv("GROQ_API_KEY")
SKILL_SETS_FILE          = "role_skill_sets.json"
OUTPUT_FILE              = Path("filtered_datasets.json")
KAGGLE_RESULTS_PER_TAG   = 10   # results fetched per tag search
MAX_TAGS_PER_SKILL       = 3    # how many tags to fan-out search per skill
MAX_DATASETS_PER_SKILL   = 25   # cap after dedup across tag searches

DS_TECHNIQUES = [
    "classification", "regression", "clustering", "nlp",
    "time_series", "recommendation", "anomaly_detection",
    "computer_vision", "deep_learning", "data_wrangling", "other"
]
DS_DOMAINS = [
    "finance", "healthcare", "retail", "hr", "marketing",
    "supply_chain", "sports", "real_estate", "education", "other"
]

PRIORITY_WEIGHT = {"critical": 3, "high": 2, "medium": 1}

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

kaggle.api.authenticate()
api = kaggle.api


# ─────────────────────────────────────────────
# LOAD TAXONOMY
# ─────────────────────────────────────────────

def load_taxonomy(path: str = SKILL_SETS_FILE) -> dict:
    """Load role_skill_sets.json. Raises clearly if file is missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"'{path}' not found. Make sure role_skill_sets.json is in the same directory."
        )
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📚  Loaded taxonomy: {len(data['roles'])} roles\n")
    return data


def get_roles_to_run(taxonomy: dict, role_filter: list[str] | None) -> list[dict]:
    """Return the subset of roles to process based on CLI filter."""
    all_roles = taxonomy["roles"]
    if not role_filter:
        return all_roles
    matched = [r for r in all_roles if r["role_id"] in role_filter]
    if not matched:
        valid = [r["role_id"] for r in all_roles]
        raise ValueError(f"No roles matched {role_filter}. Valid IDs: {valid}")
    return matched


def extract_skills_from_role(role: dict) -> list[dict]:
    """
    Flatten all skill categories for a role into a list of enriched skill objects.

    Each object has:
        skill          — display name
        category       — technical_tools / analytical_skills / etc.
        category_priority — critical / high / medium
        proficiency_expected
        kaggle_tags    — list of Kaggle-friendly search terms
    """
    skills = []
    for category_name, category_data in role["skill_categories"].items():
        cat_priority = category_data.get("priority", "medium")
        for s in category_data["skills"]:
            # Clean tags: replace underscores with spaces for Kaggle search
            raw_tags = s.get("tags", [])
            clean_tags = [t.replace("_", " ") for t in raw_tags]
            skills.append({
                "skill":                 s["skill"],
                "category":              category_name,
                "category_priority":     cat_priority,
                "proficiency_expected":  s.get("proficiency_expected", "intermediate"),
                "kaggle_tags":           clean_tags,
                "raw_tags":              raw_tags,
            })
    return skills


# ─────────────────────────────────────────────
# KAGGLE FETCH — tag-based fan-out
# ─────────────────────────────────────────────

def fetch_kaggle_datasets(skill_obj: dict) -> list[dict]:
    """
    Search Kaggle using the skill's tags instead of the raw skill name.
    Fans out across up to MAX_TAGS_PER_SKILL tags, then deduplicates.
    """
    skill_name = skill_obj["skill"]
    tags       = skill_obj["kaggle_tags"][:MAX_TAGS_PER_SKILL]

    # Fallback: if no tags, use a cleaned version of the skill name
    if not tags:
        tags = [re.sub(r'\(.*?\)', '', skill_name).strip()[:40]]

    all_raw, seen_refs = [], set()

    for tag in tags:
        try:
            print(f"      🔎  tag search: '{tag}'")
            results = api.dataset_list(search=tag, sort_by="votes")
            for ds in results[:KAGGLE_RESULTS_PER_TAG]:
                if ds.ref not in seen_refs:
                    seen_refs.add(ds.ref)
                    all_raw.append({
                        "title":        ds.title,
                        "ref":          ds.ref,
                        "url":          f"https://www.kaggle.com/datasets/{ds.ref}",
                        "description":  ds.subtitle or "",
                        "votes":        ds.vote_count or 0,
                        "downloads":    ds.download_count or 0,
                        "usability":    round(ds.usability_rating, 2) if ds.usability_rating else None,
                        "last_updated": str(ds.last_updated)[:10],
                        "tags":         [t.name for t in ds.tags] if ds.tags else [],
                        "author":       ds.creator_name or "",
                        "matched_tag":  tag,   # which tag surfaced this dataset
                    })
        except Exception as e:
            print(f"      ⚠️  Tag search failed for '{tag}': {e}")

    result = all_raw[:MAX_DATASETS_PER_SKILL]
    print(f"      → {len(result)} unique datasets after dedup\n")
    return result


# ─────────────────────────────────────────────
# GROQ HELPERS
# ─────────────────────────────────────────────

def _groq_call(prompt: str, max_tokens: int) -> str:
    """Call Groq API with retry on rate limit."""
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is not set. Export it before running.")
    for attempt in range(4):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       "llama-3.1-8b-instant",
                    "messages":    [{"role": "user", "content": prompt}],
                    "max_tokens":  max_tokens,
                    "temperature": 0,
                },
                timeout=15,
            )
            data = response.json()
            if "error" in data:
                code = data["error"].get("code", "")
                if code == "rate_limit_exceeded":
                    wait = 3 * (attempt + 1)
                    print(f"      ⏳  Rate limit — waiting {wait}s (attempt {attempt+1}/4)...")
                    time.sleep(wait)
                    continue
                print(f"      ❌  Groq error: {data['error'].get('message', data['error'])}")
                return "{}"
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"      ⚠️  Groq call error: {e}")
            time.sleep(2)
    return "{}"


def classify_dataset(title: str, description: str) -> dict:
    """
    One Groq call per dataset.
    Returns: { technique, domain, relevant (bool) }
    """
    prompt = (
        f"You are a data science dataset classifier.\n"
        f"Classify using ONLY these fixed lists:\n"
        f"Techniques: {DS_TECHNIQUES}\n"
        f"Domains: {DS_DOMAINS}\n"
        f"Title: {title}\n"
        f"Description: {description or 'N/A'}\n"
        f"Reply ONLY valid JSON (no markdown): "
        f'{{ "technique": "<value>", "domain": "<value>", "relevant": true }}\n'
        f"Set relevant=false ONLY if the dataset has no tabular/structured data at all "
        f"(e.g. raw audio, raw images only, sensor streams only)."
    )
    try:
        raw = _groq_call(prompt, max_tokens=80)
        # Strip any accidental markdown fences
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"      ⚠️  Classification parse failed for '{title}': {e}")
        return {"technique": "other", "domain": "other", "relevant": False}


# ─────────────────────────────────────────────
# FILTER — AI classify + optional domain filter
# ─────────────────────────────────────────────

def filter_datasets_for_skill(
    skill_obj: dict,
    datasets:  list[dict],
    target_domains: list[str] | None = None,
) -> list[dict]:
    """
    AI-classify each dataset, optionally filter by domain,
    and attach metadata from the skill taxonomy.
    """
    kept = []
    priority_score = PRIORITY_WEIGHT.get(skill_obj["category_priority"], 1)

    for d in datasets:
        result = classify_dataset(d.get("title", ""), d.get("description", ""))
        d["ai_technique"]    = result.get("technique", "other")
        d["ai_domain"]       = result.get("domain",    "other")
        d["relevant"]        = result.get("relevant",  True)
        d["priority_score"]  = priority_score
        d["skill_name"]      = skill_obj["skill"]
        d["skill_category"]  = skill_obj["category"]
        d["skill_priority"]  = skill_obj["category_priority"]
        d["proficiency_expected"] = skill_obj["proficiency_expected"]
        d["matched_tags"]    = skill_obj["raw_tags"]

        icon = "✅" if d["relevant"] else "⛔"
        print(f"      {icon}  [{d['ai_domain']} | {d['ai_technique']}]  {d['title']}")

        if not d["relevant"]:
            continue
        if target_domains and d["ai_domain"] not in target_domains:
            continue
        kept.append(d)

    print(f"      → kept {len(kept)}/{len(datasets)}\n")
    return kept


# ─────────────────────────────────────────────
# PER-ROLE PIPELINE
# ─────────────────────────────────────────────

def run_role(role: dict) -> tuple[str, str, dict]:
    """
    Full pipeline for one role: fetch → AI filter.
    Returns (role_id, role_name, skill_map) — caller consolidates into the global repo.
    """
    role_id   = role["role_id"]
    role_name = role["role_name"]

    print(f"\n{'='*60}")
    print(f"🎯  Role: {role_name}  ({role_id})")
    print(f"{'='*60}\n")

    skills = extract_skills_from_role(role)
    print(f"📋  {len(skills)} skills to process\n")

    all_filtered: dict[str, dict] = {}

    for i, skill_obj in enumerate(skills, 1):
        skill_name = skill_obj["skill"]
        print(f"  [{i}/{len(skills)}] {skill_name}  "
              f"[{skill_obj['category']} | {skill_obj['category_priority']}]")

        datasets = fetch_kaggle_datasets(skill_obj)
        if not datasets:
            print("      ⚠️  No datasets found, skipping.\n")
            continue

        filtered = filter_datasets_for_skill(skill_obj, datasets, target_domains=None)

        all_filtered[skill_name] = {
            "skill_meta": {
                "skill":                skill_name,
                "category":             skill_obj["category"],
                "category_priority":    skill_obj["category_priority"],
                "proficiency_expected": skill_obj["proficiency_expected"],
                "kaggle_tags":          skill_obj["kaggle_tags"],
                "raw_tags":             skill_obj["raw_tags"],
            },
            "datasets": filtered,
        }

    total_datasets = sum(len(v["datasets"]) for v in all_filtered.values())
    print(f"\n    {len(all_filtered)} skills | {total_datasets} filtered datasets\n")
    return role_id, role_name, all_filtered


def merge_role_into_repo(repo: dict, role_id: str, role_name: str, skill_map: dict) -> None:
    """Consolidate one role's datasets into the global repo dict (deduplicated by URL)."""
    for skill_name, skill_data in skill_map.items():
        skill_meta = skill_data["skill_meta"]
        ref_base = {
            "role_id":              role_id,
            "role_name":            role_name,
            "skill":                skill_name,
            "category":             skill_meta["category"],
            "category_priority":    skill_meta["category_priority"],
            "proficiency_expected": skill_meta["proficiency_expected"],
        }
        for ds in skill_data["datasets"]:
            url = ds.get("url", "")
            if not url:
                continue
            if url not in repo["datasets"]:
                repo["datasets"][url] = {
                    "title":            ds.get("title", ""),
                    "url":              url,
                    "description":      ds.get("description", ""),
                    "votes":            ds.get("votes", 0),
                    "downloads":        ds.get("downloads", 0),
                    "usability":        ds.get("usability"),
                    "tags":             ds.get("tags", []),
                    "ai_domain":        ds.get("ai_domain", "other"),
                    "ai_technique":     ds.get("ai_technique", "other"),
                    "referenced_by":    [ref_base],
                    "cross_role_count": 1,
                }
            else:
                existing = repo["datasets"][url]["referenced_by"]
                if not any(r["role_id"] == role_id and r["skill"] == skill_name for r in existing):
                    existing.append(ref_base)
                    repo["datasets"][url]["cross_role_count"] = len(
                        set(r["role_id"] for r in existing)
                    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch + AI-filter Kaggle datasets for all roles in the skill taxonomy."
    )
    parser.add_argument(
        "--role", nargs="*", default=None,
        help=(
            "One or more role_ids to process. "
            "If omitted, all 8 roles are run. "
            "Example: --role data_scientist financial_analyst"
        ),
    )
    args = parser.parse_args()

    taxonomy     = load_taxonomy(SKILL_SETS_FILE)
    roles_to_run = get_roles_to_run(taxonomy, args.role)

    print(f" Running pipeline for {len(roles_to_run)} role(s): "
          f"{[r['role_id'] for r in roles_to_run]}\n")

    repo: dict = {
        "generated":             "",
        "roles_included":        [],
        "total_unique_datasets": 0,
        "datasets":              {},
    }

    for role in roles_to_run:
        role_id, role_name, skill_map = run_role(role)
        repo["roles_included"].append({"role_id": role_id, "role_name": role_name})
        merge_role_into_repo(repo, role_id, role_name, skill_map)

    repo["total_unique_datasets"] = len(repo["datasets"])
    repo["generated"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(repo, f, indent=2, ensure_ascii=False)

    print("\n  All done!")
    print(f"  {repo['total_unique_datasets']} unique datasets saved to: {OUTPUT_FILE}")
    print("  Next step: run rank_datasets.py to score and produce dataset_lookup.json\n")


if __name__ == "__main__":
    main()
