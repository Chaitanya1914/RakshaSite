"""
Milestone 4: Real-Data Ablation Study Script
RakshaSite Multi-Agent Pipeline Benchmark across 50 Authentic Worksite Photographs.

Evaluates 3 architectural tiers:
  Tier 1: `yolo_only` (Raw YOLOv8n detector output without containment reasoning or statutory RAG)
  Tier 2: `yolo_plus_rules` (YOLOv8 + spatial containment rules + static heuristic lookup)
  Tier 3: `full_agent_pipeline` (Full 4-Agent LangGraph: Temporal Gate + XGBoost + ChromaDB RAG + Report Agent)

Outputs: `reports/ablation_study.json`
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.pipeline import run_ablation_study


def main():
    parser = argparse.ArgumentParser(description="Run 3-tier ablation study on real photographs.")
    parser.add_argument(
        "--output",
        default="reports/ablation_study.json",
        help="Path to output ablation results JSON",
    )
    parser.add_argument(
        "--test-cases",
        default="tests/test_cases.json",
        help="Path to 50 real validation photographs test cases JSON",
    )
    args = parser.parse_args()

    print(f"Running 3-tier ablation study on test cases from: {args.test_cases}")
    results = run_ablation_study(output_path=args.output, test_cases_path=args.test_cases)

    print("\n" + "=" * 80)
    print("RAKSHASITE REAL-DATA 3-TIER ABLATION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Tier':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'FPR':<8} | {'Sev. Acc':<10} | {'Cit. Acc':<10}")
    print("-" * 80)
    for tier, metrics in results.items():
        print(
            f"{tier:<22} | "
            f"{metrics['precision']:>8.2f}% | "
            f"{metrics['recall']:>8.2f}% | "
            f"{metrics['f1_score']:>8.2f}% | "
            f"{metrics['false_positive_rate']:>6.2f}% | "
            f"{metrics['severity_accuracy']:>8.2f}% | "
            f"{metrics['section_citation_accuracy']:>8.2f}%"
        )
    print("=" * 80)
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
