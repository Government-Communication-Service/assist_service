#!/usr/bin/env python3
"""
Compare AI findings with ground truth violations and generate a detailed report.
"""
import json
import sys
from typing import List, Set


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def normalize_rule_name(rule: str) -> str:
    """Normalize rule names for comparison (case-insensitive)."""
    return rule.strip().lower()


def get_document_rules(rules: List[dict]) -> Set[str]:
    """Extract set of rule names from rules_broken list."""
    return {normalize_rule_name(r.get('rule_name', r.get('rule', ''))) for r in rules}


def calculate_metrics(true_positives: int, false_positives: int, false_negatives: int) -> dict:
    """Calculate precision, recall, and F1 score."""
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1_score, 3)
    }


def compare_documents(ground_truth: dict, ai_findings: dict, limit: int = None) -> dict:
    """Compare AI findings with ground truth for all documents.

    Args:
        ground_truth: Ground truth violations data
        ai_findings: AI predicted violations data
        limit: Maximum number of documents to analyze (None for all)
    """

    ground_truth_docs = ground_truth['document_violations']
    ai_docs = ai_findings['document_violations']

    report = {
        "metadata": {
            "total_documents": len(ground_truth_docs),
            "documents_analyzed": 0
        },
        "overall_metrics": {},
        "per_document_analysis": {},
        "per_rule_analysis": {},
        "false_positives_summary": {},
        "false_negatives_summary": {},
        "detailed_comparison": {}
    }

    # Track overall stats
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    # Track per-rule stats
    rule_stats = {}

    # Compare each document
    doc_count = 0
    for doc_name, ground_truth_data in ground_truth_docs.items():
        if limit and doc_count >= limit:
            break
        doc_count += 1
        # Find corresponding AI findings
        if doc_name not in ai_docs:
            # AI didn't analyze this document at all
            continue

        ai_data = ai_docs[doc_name]

        report['metadata']['documents_analyzed'] += 1

        # Get rule sets from rules_broken (authoritative list of all violations)
        ground_truth_rules = get_document_rules(ground_truth_data['rules_broken'])
        ai_rules = get_document_rules(ai_data['rules_broken'])

        # Calculate true positives, false positives, false negatives
        true_positives = ground_truth_rules & ai_rules
        false_positives = ai_rules - ground_truth_rules
        false_negatives = ground_truth_rules - ai_rules

        tp_count = len(true_positives)
        fp_count = len(false_positives)
        fn_count = len(false_negatives)

        # Update overall counts
        overall_tp += tp_count
        overall_fp += fp_count
        overall_fn += fn_count

        # Calculate metrics for this document
        doc_metrics = calculate_metrics(tp_count, fp_count, fn_count)

        # Store document analysis
        report['per_document_analysis'][doc_name] = {
            "true_positives": sorted(true_positives),
            "false_positives": sorted(false_positives),
            "false_negatives": sorted(false_negatives),
            "counts": {
                "true_positives": tp_count,
                "false_positives": fp_count,
                "false_negatives": fn_count,
                "ground_truth_violations": len(ground_truth_rules),
                "ai_detected_violations": len(ai_rules)
            },
            "metrics": doc_metrics
        }

        # Track per-rule statistics
        for rule in true_positives:
            if rule not in rule_stats:
                rule_stats[rule] = {"tp": 0, "fp": 0, "fn": 0}
            rule_stats[rule]["tp"] += 1

        for rule in false_positives:
            if rule not in rule_stats:
                rule_stats[rule] = {"tp": 0, "fp": 0, "fn": 0}
            rule_stats[rule]["fp"] += 1

        for rule in false_negatives:
            if rule not in rule_stats:
                rule_stats[rule] = {"tp": 0, "fp": 0, "fn": 0}
            rule_stats[rule]["fn"] += 1

        # Detailed comparison with confidence scores
        detailed = {
            "ground_truth": {
                "rules_broken": ground_truth_data['rules_broken'],
                "rule_count": len(ground_truth_data['rules_broken'])
            },
            "ai_findings": {
                "rules_found": ai_data['rules_broken'],
                "rule_count": len(ai_data['rules_broken'])
            },
            "comparison": {
                "correctly_identified": sorted(true_positives),
                "incorrectly_flagged": sorted(false_positives),
                "missed_violations": sorted(false_negatives)
            }
        }

        # Add confidence scores for each violation type
        ai_violations_by_rule = {}
        for v in ai_data['violations']:
            rule_norm = normalize_rule_name(v.get('rule_name', v.get('rule', '')))
            ai_violations_by_rule[rule_norm] = v

        report['detailed_comparison'][doc_name] = detailed

    # Calculate overall metrics
    report['overall_metrics'] = {
        "total_true_positives": overall_tp,
        "total_false_positives": overall_fp,
        "total_false_negatives": overall_fn,
        "detection_rate": (
            round(overall_tp / (overall_tp + overall_fn) * 100, 2)
            if (overall_tp + overall_fn) > 0
            else 0
        ),
        **calculate_metrics(overall_tp, overall_fp, overall_fn)
    }

    # Per-rule analysis
    for rule, stats in rule_stats.items():
        rule_metrics = calculate_metrics(stats['tp'], stats['fp'], stats['fn'])
        report['per_rule_analysis'][rule] = {
            "true_positives": stats['tp'],
            "false_positives": stats['fp'],
            "false_negatives": stats['fn'],
            "detection_rate": (
                round(stats['tp'] / (stats['tp'] + stats['fn']) * 100, 2)
                if (stats['tp'] + stats['fn']) > 0
                else 0
            ),
            **rule_metrics
        }

    # Summarize false positives and false negatives
    fp_summary = {}
    fn_summary = {}

    for doc_name, analysis in report['per_document_analysis'].items():
        for fp in analysis['false_positives']:
            if fp not in fp_summary:
                fp_summary[fp] = []
            fp_summary[fp].append(doc_name)

        for fn in analysis['false_negatives']:
            if fn not in fn_summary:
                fn_summary[fn] = []
            fn_summary[fn].append(doc_name)

    report['false_positives_summary'] = {
        rule: {"count": len(docs), "documents": docs}
        for rule, docs in sorted(fp_summary.items(), key=lambda x: len(x[1]), reverse=True)
    }

    report['false_negatives_summary'] = {
        rule: {"count": len(docs), "documents": docs}
        for rule, docs in sorted(fn_summary.items(), key=lambda x: len(x[1]), reverse=True)
    }

    # Best and worst performing rules
    sorted_rules = sorted(
        report['per_rule_analysis'].items(),
        key=lambda x: x[1]['f1_score'],
        reverse=True
    )

    report['performance_highlights'] = {
        "best_performing_rules": [
            {"rule": rule, **metrics}
            for rule, metrics in sorted_rules[:5]
        ],
        "worst_performing_rules": [
            {"rule": rule, **metrics}
            for rule, metrics in sorted_rules[-5:]
        ]
    }

    return report


def main():
    """Main execution."""
    # Parse command line arguments
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            print(f"Analyzing first {limit} documents...")
        except ValueError:
            print(f"Invalid limit: {sys.argv[1]}. Using all documents.")

    print("Loading data...")

    ground_truth = load_json('ground_truth.json')
    ai_findings = load_json('outputs/ai_predicted_violations.json')

    print("Comparing AI findings with ground truth...")

    report = compare_documents(ground_truth, ai_findings, limit=limit)

    # Save report
    output_file = 'outputs/ai_performance_report.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print("AI PERFORMANCE REPORT")
    print(f"{'='*70}")

    print(f"\nDocuments Analyzed: {report['metadata']['documents_analyzed']}/{report['metadata']['total_documents']}")

    print(f"\n{'-'*70}")
    print("OVERALL METRICS")
    print(f"{'-'*70}")
    metrics = report['overall_metrics']
    print(f"True Positives:  {metrics['total_true_positives']}")
    print(f"False Positives: {metrics['total_false_positives']}")
    print(f"False Negatives: {metrics['total_false_negatives']}")
    print(f"\nDetection Rate:  {metrics['detection_rate']}%")
    print(f"Precision:       {metrics['precision']:.1%}")
    print(f"Recall:          {metrics['recall']:.1%}")
    print(f"F1 Score:        {metrics['f1_score']:.3f}")

    print(f"\n{'-'*70}")
    print("TOP 5 BEST PERFORMING RULES")
    print(f"{'-'*70}")
    for item in report['performance_highlights']['best_performing_rules']:
        print(f"{item['rule']:.<50} F1: {item['f1_score']:.3f}")

    print(f"\n{'-'*70}")
    print("TOP 5 WORST PERFORMING RULES")
    print(f"{'-'*70}")
    for item in report['performance_highlights']['worst_performing_rules']:
        print(f"{item['rule']:.<50} F1: {item['f1_score']:.3f}")

    if report['false_positives_summary']:
        print(f"\n{'-'*70}")
        print("MOST COMMON FALSE POSITIVES")
        print(f"{'-'*70}")
        for rule, data in list(report['false_positives_summary'].items())[:5]:
            print(f"{rule}: {data['count']} occurrences")

    if report['false_negatives_summary']:
        print(f"\n{'-'*70}")
        print("MOST COMMON FALSE NEGATIVES (MISSED)")
        print(f"{'-'*70}")
        for rule, data in list(report['false_negatives_summary'].items())[:5]:
            print(f"{rule}: {data['count']} occurrences")

    print(f"\n{'-'*70}")
    print(f"Full report saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
