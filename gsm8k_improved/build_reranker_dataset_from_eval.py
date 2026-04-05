from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a reranker dataset from eval rows with saved candidates."
    )
    parser.add_argument(
        "--run-summary",
        required=True,
        help="Path to a run_summary.json that contains eval rows with candidates.",
    )
    parser.add_argument(
        "--section",
        default="eval_after",
        help="Evaluation section to read, e.g. eval_after or eval_before.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for pairwise/listwise reranker data.",
    )
    return parser


def normalize_answer(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    matches = re.findall(r"-?\d+", text.replace(",", ""))
    return matches[-1] if matches else ""


def candidate_label(candidate: dict[str, Any], gold_answer: str) -> int:
    predicted = normalize_answer(candidate.get("predicted_answer", ""))
    return int(predicted == gold_answer)


def candidate_quality(candidate: dict[str, Any], gold_answer: str) -> float:
    predicted = normalize_answer(candidate.get("predicted_answer", ""))
    exact = predicted == gold_answer and bool(candidate.get("has_strict_xml"))
    if exact:
        return 1000.0 + float(candidate.get("rerank_score", 0.0))
    if predicted == gold_answer:
        return 700.0 + float(candidate.get("rerank_score", 0.0))
    return float(candidate.get("rerank_score", 0.0))


def main() -> int:
    args = build_parser().parse_args()
    run_summary = Path(args.run_summary).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    obj = json.loads(run_summary.read_text(encoding="utf-8"))
    section = obj[args.section]
    rows = section["rows"]

    pairwise_rows: list[dict[str, Any]] = []
    listwise_rows: list[dict[str, Any]] = []
    hard_case_count = 0

    for row in rows:
        candidates = row.get("candidates")
        if not candidates:
            continue
        gold_answer = normalize_answer(row.get("gold_answer", ""))
        question = str(row["question"])
        enriched = []
        for candidate in candidates:
            enriched_candidate = dict(candidate)
            enriched_candidate["label"] = candidate_label(candidate, gold_answer)
            enriched_candidate["quality_score"] = candidate_quality(candidate, gold_answer)
            enriched.append(enriched_candidate)

        sorted_candidates = sorted(enriched, key=lambda item: item["quality_score"], reverse=True)
        listwise_rows.append(
            {
                "question": question,
                "gold_answer": gold_answer,
                "candidates": sorted_candidates,
            }
        )

        chosen = max(enriched, key=lambda item: float(item.get("rerank_score", 0.0)))
        gold_candidates = [candidate for candidate in enriched if candidate["label"] == 1]
        wrong_candidates = [candidate for candidate in enriched if candidate["label"] == 0]
        if gold_candidates and chosen["label"] == 0:
            hard_case_count += 1
            best_gold = max(gold_candidates, key=lambda item: item["quality_score"])
            pairwise_rows.append(
                {
                    "question": question,
                    "gold_answer": gold_answer,
                    "chosen": best_gold["text"],
                    "rejected": chosen["text"],
                    "chosen_answer": best_gold["predicted_answer"],
                    "rejected_answer": chosen["predicted_answer"],
                    "chosen_rerank_score": best_gold.get("rerank_score"),
                    "rejected_rerank_score": chosen.get("rerank_score"),
                    "chosen_verifier_score": best_gold.get("verifier_score"),
                    "rejected_verifier_score": chosen.get("verifier_score"),
                    "pair_type": "gold_available_but_not_selected",
                }
            )
        elif gold_candidates and wrong_candidates:
            best_gold = max(gold_candidates, key=lambda item: item["quality_score"])
            strongest_wrong = max(wrong_candidates, key=lambda item: item["quality_score"])
            pairwise_rows.append(
                {
                    "question": question,
                    "gold_answer": gold_answer,
                    "chosen": best_gold["text"],
                    "rejected": strongest_wrong["text"],
                    "chosen_answer": best_gold["predicted_answer"],
                    "rejected_answer": strongest_wrong["predicted_answer"],
                    "chosen_rerank_score": best_gold.get("rerank_score"),
                    "rejected_rerank_score": strongest_wrong.get("rerank_score"),
                    "chosen_verifier_score": best_gold.get("verifier_score"),
                    "rejected_verifier_score": strongest_wrong.get("verifier_score"),
                    "pair_type": "gold_vs_best_wrong",
                }
            )

    pairwise_path = output_dir / "pairwise_reranker_train.jsonl"
    with pairwise_path.open("w", encoding="utf-8") as fh:
        for row in pairwise_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    listwise_path = output_dir / "listwise_reranker_train.jsonl"
    with listwise_path.open("w", encoding="utf-8") as fh:
        for row in listwise_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "run_summary": str(run_summary),
        "section": args.section,
        "rows_with_candidates": len(listwise_rows),
        "pairwise_rows": len(pairwise_rows),
        "hard_case_count": hard_case_count,
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
