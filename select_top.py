"""
Select top N datasets from dataset_lookup.json with guaranteed skill coverage.

Strategy:
  1. Iterate all unique skills from role_skill_sets.json in priority order
     (critical first, then high, then medium).
  2. For each skill not yet covered by an already-selected dataset, pick
     the highest-scoring dataset whose tags contain that skill.
  3. Fill any remaining slots up to N with the next-best unselected datasets
     by final_score.

A single dataset can cover many skills, so 50 datasets typically cover
the full critical and high-priority skill list.

Usage:
    python select_top.py
    python select_top.py --n 50 --input dataset_lookup.json --output top50.json
"""

import json
import argparse
from pathlib import Path


PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2}

DEFAULT_INPUT    = Path("dataset_lookup.json")
DEFAULT_SKILLS   = Path("role_skill_sets.json")
DEFAULT_OUTPUT   = Path("top50.json")
DEFAULT_N        = 50


def load_json(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"'{path}' not found.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_skills(taxonomy: dict) -> list[tuple[str, str]]:
    """
    Return list of (skill_name, priority) sorted by priority descending.
    If a skill appears in multiple roles, keep its highest priority.
    """
    skills: dict[str, str] = {}
    for role in taxonomy["roles"]:
        for cat_name, cat in role["skill_categories"].items():
            prio = cat.get("priority", "medium")
            for s in cat["skills"]:
                name = s["skill"]
                current = PRIORITY_ORDER.get(skills.get(name, "medium"), 2)
                incoming = PRIORITY_ORDER.get(prio, 2)
                if incoming < current:  # lower index = higher priority
                    skills[name] = prio

    return sorted(skills.items(), key=lambda x: PRIORITY_ORDER.get(x[1], 2))


def build_skill_index(datasets: list[dict]) -> dict[str, list[int]]:
    """Map each tag value to the list of dataset indices that carry it."""
    index: dict[str, list[int]] = {}
    for i, ds in enumerate(datasets):
        for tag in ds.get("tags", []):
            index.setdefault(tag, []).append(i)
    return index


def select(datasets: list[dict], skills: list[tuple[str, str]], n: int) -> list[dict]:
    """
    Greedy set-cover then fill.

    datasets  — already sorted by final_score descending
    skills    — (skill_name, priority) sorted by priority descending
    n         — target count
    """
    skill_index = build_skill_index(datasets)
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    covered_skills: set[str] = set()

    # --- pass 1: coverage ---
    for skill_name, priority in skills:
        if len(selected_indices) >= n:
            break
        if skill_name in covered_skills:
            continue

        candidates = skill_index.get(skill_name, [])
        # candidates are in insertion order; since datasets is sorted by score,
        # earlier indices are higher-scored — pick the first unselected one
        for idx in candidates:
            if idx not in selected_set:
                selected_indices.append(idx)
                selected_set.add(idx)
                # mark all skills covered by this dataset
                for tag in datasets[idx].get("tags", []):
                    covered_skills.add(tag)
                break

    # --- pass 2: fill remaining slots with global top ---
    for idx, ds in enumerate(datasets):
        if len(selected_indices) >= n:
            break
        if idx not in selected_set:
            selected_indices.append(idx)
            selected_set.add(idx)

    # preserve final_score order in output
    result = sorted(selected_indices, key=lambda i: datasets[i]["final_score"], reverse=True)
    return [datasets[i] for i in result]


def coverage_report(selected: list[dict], skills: list[tuple[str, str]]) -> None:
    all_tags: set[str] = set()
    for ds in selected:
        all_tags.update(ds.get("tags", []))

    uncovered = [(s, p) for s, p in skills if s not in all_tags]
    covered_count = len(skills) - len(uncovered)
    print(f"  Skills covered : {covered_count}/{len(skills)}")

    by_prio: dict[str, list[str]] = {}
    for s, p in uncovered:
        by_prio.setdefault(p, []).append(s)

    if not uncovered:
        print("  All skills covered.")
    else:
        print(f"  Uncovered      : {len(uncovered)}")
        for prio in ["critical", "high", "medium"]:
            if prio in by_prio:
                print(f"    [{prio}]")
                for s in by_prio[prio]:
                    print(f"      - {s}")


def main():
    parser = argparse.ArgumentParser(
        description="Select top N datasets with full skill coverage from dataset_lookup.json"
    )
    parser.add_argument("--input",  type=Path, default=DEFAULT_INPUT,
                        help=f"Ranked lookup file (default: {DEFAULT_INPUT})")
    parser.add_argument("--skills", type=Path, default=DEFAULT_SKILLS,
                        help=f"Skills taxonomy (default: {DEFAULT_SKILLS})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--n",      type=int,  default=DEFAULT_N,
                        help=f"Number of datasets to select (default: {DEFAULT_N})")
    args = parser.parse_args()

    datasets = load_json(args.input)
    taxonomy = load_json(args.skills)

    skills = extract_skills(taxonomy)
    print(f"Loaded {len(datasets)} datasets, {len(skills)} unique skills")
    print(f"Selecting top {args.n} with coverage pass...\n")

    selected = select(datasets, skills, args.n)

    print(f"Selected: {len(selected)} datasets\n")
    coverage_report(selected, skills)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
