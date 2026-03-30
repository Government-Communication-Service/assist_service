#!/usr/bin/env python3
"""
Generate predicted style guide violations using the local style_guide_checker.
This script analyzes all golden dataset documents against GOV.UK style guide rules.
"""
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_ai_prediction(violations: List[Dict]) -> Dict:
    """
    Convert style_guide_checker violations list into the ai_predicted_violations.json shape.
    """
    rules_broken = []
    seen_rule_names = set()

    for violation in violations:
        rule_id = violation.get("rule_id")
        rule_name = violation.get("rule_title") or violation.get("rule_name")
        # Deduplicate by normalised rule name so that b-suffix variants of the
        # same rule (e.g. rule_138 and rule_138b) don't produce duplicate entries.
        # Use the base rule_id (strip trailing 'b') for the stored record.
        if rule_id and rule_name and rule_name.lower() not in seen_rule_names:
            seen_rule_names.add(rule_name.lower())
            base_id = rule_id[:-1] if rule_id.endswith("b") else rule_id
            rules_broken.append({
                "rule_id": base_id,
                "rule_name": rule_name
            })

    formatted_violations = []
    seen_violation_names = set()
    for violation in violations:
        rule_id = violation.get("rule_id")
        rule_name = violation.get("rule_title") or violation.get("rule_name")
        sentences = violation.get("sentences", [])
        if rule_id and rule_name and rule_name.lower() not in seen_violation_names:
            seen_violation_names.add(rule_name.lower())
            base_id = rule_id[:-1] if rule_id.endswith("b") else rule_id
            formatted_violations.append({
                "rule_id": base_id,
                "rule_name": rule_name,
                "occurrences": sentences
            })

    return {
        "rules_broken": rules_broken,
        "violations": formatted_violations
    }


def run_style_guide_checker(
    checker_path: Path,
    document_name: str,
    output_path: Path,
    working_dir: Path
) -> List[Dict]:
    """Run style_guide_checker.py for a document and return parsed violations list."""
    command = [
        sys.executable,
        str(checker_path),
        "--document",
        document_name,
        "--output",
        str(output_path)
    ]

    result = subprocess.run(
        command,
        cwd=str(working_dir),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"style_guide_checker failed for {document_name}: {result.stderr.strip()}"
        )

    if not output_path.exists():
        raise FileNotFoundError(f"Expected output not found: {output_path}")

    with open(output_path, "r", encoding="utf-8") as f:
        violations = json.load(f)

    return violations


def generate_ai_violations(
    test_docs_dir: str,
    checker_path: str,
    output_file: str
):
    """
    Generate AI violations by running style_guide_checker.py over all test documents.
    """
    start_time = time.time()
    logger.info("Starting AI violations generation via style_guide_checker")

    test_docs_path = Path(test_docs_dir)
    checker_file = Path(checker_path)
    working_dir = checker_file.parent

    document_files = sorted(
        f for f in test_docs_path.glob("*.txt")
        if not f.stem.endswith("_FIXED")
    )
    logger.info(f"Found {len(document_files)} documents to analyze")

    output = {
        "ai_analysis_metadata": {
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "total_documents_analyzed": len(document_files),
            "note": "AI-generated analysis using style_guide_checker"
        },
        "document_violations": {}
    }

    total_violations = 0
    temp_output_dir = Path("/tmp/style_guide_checker_outputs")
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    for i, doc_file in enumerate(document_files, 1):
        doc_name = doc_file.name
        logger.info(f"Processing {i}/{len(document_files)}: {doc_name}")

        output_path = temp_output_dir / f"{doc_name}.json"
        try:
            violations = run_style_guide_checker(
                checker_file,
                doc_name,
                output_path,
                working_dir
            )
            analysis = build_ai_prediction(violations)
            total_violations += len(analysis.get("violations", []))
            output["document_violations"][doc_name] = analysis
        except Exception as exc:
            logger.error(f"Error processing {doc_name}: {exc}")
            output["document_violations"][doc_name] = {
                "rules_broken": [],
                "violations": [],
                "error": str(exc)
            }

    elapsed_time = time.time() - start_time
    output["ai_analysis_metadata"].update({
        "total_violations": total_violations,
        "elapsed_time_seconds": round(elapsed_time, 2),
        "elapsed_time_formatted": f"{int(elapsed_time // 60)}m {int(elapsed_time % 60)}s"
    })

    logger.info(f"Saving results to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    logger.info("Analysis complete!")
    print("=" * 70)
    logger.info(f"   Documents analyzed: {len(document_files)}")
    logger.info(f"   Total violations found: {total_violations}")
    logger.info(f"   Time: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s ({elapsed_time:.2f}s)")
    logger.info(f"   Results saved to: {output_file}")
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    test_docs_dir = script_dir.parent / "golden_dataset"
    # style_guide_checker.py lives in app/style_guide/, not in scripts/
    checker_path = script_dir.parent.parent.parent / "app" / "style_guide" / "style_guide_checker.py"
    outputs_dir = script_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    output_file = outputs_dir / "ai_predicted_violations.json"

    generate_ai_violations(
        str(test_docs_dir),
        str(checker_path),
        str(output_file)
    )


if __name__ == "__main__":
    main()
