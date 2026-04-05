from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Augment verifier dataset with targeted failure cases from eval diffs."
    )
    parser.add_argument(
        "--base-dataset-dir",
        default="/home/user/图片/gsm8k_improved/verifier_dataset_v2_strictxml",
        help="Directory containing base pointwise/pairwise verifier jsonl files.",
    )
    parser.add_argument(
        "--baseline-summary",
        default="/home/user/图片/gsm8k_improved/ngen6_evalonly_baseline30_20260403/run_summary.json",
        help="Run summary for the stronger baseline.",
    )
    parser.add_argument(
        "--candidate-summary",
        default="/home/user/图片/gsm8k_improved/evalonly_v6pairwise_aug_c2_rerankbonus30_20260403/run_summary.json",
        help="Run summary for the weaker candidate policy.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/user/图片/gsm8k_improved/verifier_dataset_v3_failure_targeted",
        help="Directory for the augmented output dataset.",
    )
    parser.add_argument(
        "--failure-dup-factor",
        type=int,
        default=16,
        help="How many extra copies to add for each targeted failure pair/row.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def build_failure_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    failure_dup_factor: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    targeted_pointwise: list[dict[str, Any]] = []
    targeted_pairwise: list[dict[str, Any]] = []
    failure_report: list[dict[str, Any]] = []

    for index, (base_row, cand_row) in enumerate(zip(baseline_rows, candidate_rows)):
        if bool(base_row.get("exact_match")) and not bool(cand_row.get("exact_match")):
            question = str(base_row["question"])
            gold_answer = str(base_row["gold_answer"])
            base_point = {
                "question": question,
                "gold_answer": gold_answer,
                "completion": base_row["raw_completion"],
                "predicted_answer": base_row["predicted_answer"],
                "label": 1,
                "quality_score": 1100.0,
                "exact_match": True,
                "abs_error": 0,
                "has_answer_tag": bool(base_row.get("has_answer_tag")),
                "has_strict_xml": bool(base_row.get("has_strict_xml")),
                "is_numeric_answer": bool(base_row.get("is_numeric_answer")),
                "source_run": "targeted_failure_baseline",
                "augmentation_tag": "targeted_failure_positive",
            }
            cand_abs_error = cand_row.get("abs_error")
            cand_point = {
                "question": question,
                "gold_answer": gold_answer,
                "completion": cand_row["raw_completion"],
                "predicted_answer": cand_row["predicted_answer"],
                "label": 0,
                "quality_score": -50.0,
                "exact_match": False,
                "abs_error": cand_abs_error,
                "has_answer_tag": bool(cand_row.get("has_answer_tag")),
                "has_strict_xml": bool(cand_row.get("has_strict_xml")),
                "is_numeric_answer": bool(cand_row.get("is_numeric_answer")),
                "source_run": "targeted_failure_candidate",
                "augmentation_tag": "targeted_failure_negative",
            }
            for _ in range(failure_dup_factor):
                targeted_pointwise.append(dict(base_point))
                targeted_pointwise.append(dict(cand_point))
                targeted_pairwise.append(
                    {
                        "question": question,
                        "gold_answer": gold_answer,
                        "chosen": base_row["raw_completion"],
                        "rejected": cand_row["raw_completion"],
                        "chosen_answer": base_row["predicted_answer"],
                        "rejected_answer": cand_row["predicted_answer"],
                        "chosen_source": "targeted_failure_baseline",
                        "rejected_source": "targeted_failure_candidate",
                        "chosen_exact_match": True,
                        "rejected_exact_match": False,
                        "chosen_abs_error": 0,
                        "rejected_abs_error": cand_abs_error,
                        "preference_margin": 1150.0,
                        "pair_type": "targeted_online_failure_exact_vs_wrong",
                    }
                )
            failure_report.append(
                {
                    "index": index,
                    "question": question,
                    "baseline_answer": base_row["predicted_answer"],
                    "candidate_answer": cand_row["predicted_answer"],
                    "candidate_abs_error": cand_abs_error,
                }
            )
    return targeted_pointwise, targeted_pairwise, failure_report


def main() -> int:
    args = build_parser().parse_args()
    base_dir = Path(args.base_dataset_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pointwise_rows = load_jsonl(base_dir / "pointwise_verifier_train.jsonl")
    pairwise_rows = load_jsonl(base_dir / "pairwise_verifier_train.jsonl")

    baseline_summary = load_json(Path(args.baseline_summary).expanduser())
    candidate_summary = load_json(Path(args.candidate_summary).expanduser())
    baseline_rows = baseline_summary["eval_after"]["rows"]
    candidate_rows = candidate_summary["eval_after"]["rows"]

    targeted_pointwise, targeted_pairwise, failure_report = build_failure_rows(
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        failure_dup_factor=args.failure_dup_factor,
    )

    pointwise_out = pointwise_rows + targeted_pointwise
    pairwise_out = pairwise_rows + targeted_pairwise
    write_jsonl(output_dir / "pointwise_verifier_train.jsonl", pointwise_out)
    write_jsonl(output_dir / "pairwise_verifier_train.jsonl", pairwise_out)

    report = {
        "input_pointwise_rows": len(pointwise_rows),
        "output_pointwise_rows": len(pointwise_out),
        "input_pairwise_rows": len(pairwise_rows),
        "output_pairwise_rows": len(pairwise_out),
        "targeted_failure_count": len(failure_report),
        "failure_dup_factor": args.failure_dup_factor,
        "targeted_failure_examples": failure_report,
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
