#!/usr/bin/env python3
"""
Analyse false positives and false negatives across the golden dataset.

Reads the latest predicted violations and ground truth, then prints
a per-document breakdown of what the checker got wrong.  Useful for
identifying patterns that need rule or ground-truth adjustments.

Usage (from the evaluation/ directory):
    python analyse_results.py
"""
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    script_dir = Path(__file__).parent
    predicted = load_json(script_dir / "outputs" / "ai_predicted_violations.json")
    ground_truth = load_json(script_dir / "ground_truth.json")

    docs_predicted = predicted["document_violations"]

    # Build normalised ground-truth rule sets keyed by document name
    gt_per_doc: dict[str, set[str]] = {
        doc: {r["rule_name"].lower() for r in data.get("rules_broken", [])}
        for doc, data in ground_truth["document_violations"].items()
    }

    # ------------------------------------------------------------------ #
    # FALSE POSITIVES — checker flagged a rule that isn't in ground truth  #
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("FALSE POSITIVES  (checker found it, ground truth doesn't have it)")
    print("=" * 70)

    any_fp = False
    for doc in sorted(docs_predicted):
        gt_rules = gt_per_doc.get(doc, set())
        for violation in docs_predicted[doc].get("violations", []):
            rule = violation.get("rule_name", "")
            if rule.lower() not in gt_rules:
                any_fp = True
                occurrences = violation.get("occurrences", [])
                print(f"\n  {doc}  |  [{rule}]")
                for occurrence in occurrences[:3]:
                    print(f"    -> {occurrence}")

    if not any_fp:
        print("  (none)")

    # ------------------------------------------------------------------ #
    # FALSE NEGATIVES — ground truth rule that the checker missed           #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("FALSE NEGATIVES  (ground truth has it, checker missed it)")
    print("=" * 70)

    any_fn = False
    for doc in sorted(gt_per_doc):
        gt_rules = gt_per_doc[doc]
        ai_rules = {
            v.get("rule_name", "").lower()
            for v in docs_predicted.get(doc, {}).get("violations", [])
        }
        missed = gt_rules - ai_rules
        if missed:
            any_fn = True
            print(f"\n  {doc}  |  missed: {missed}")

    if not any_fn:
        print("  (none)")


if __name__ == "__main__":
    main()
