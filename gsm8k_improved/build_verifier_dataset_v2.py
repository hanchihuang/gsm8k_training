from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an augmented verifier dataset with strict-XML hard negatives."
    )
    parser.add_argument(
        "--input-dir",
        default="/home/user/图片/gsm8k_improved/verifier_dataset_v1",
        help="Directory containing pointwise_verifier_train.jsonl and pairwise_verifier_train.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/user/图片/gsm8k_improved/verifier_dataset_v2_strictxml",
        help="Directory for the augmented verifier dataset.",
    )
    parser.add_argument(
        "--close-error-threshold",
        type=int,
        default=5,
        help="Maximum absolute error considered a close wrong answer.",
    )
    parser.add_argument(
        "--medium-error-threshold",
        type=int,
        default=20,
        help="Maximum absolute error considered a medium wrong answer.",
    )
    parser.add_argument(
        "--duplicate-close-wrong",
        type=int,
        default=2,
        help="Extra copies for strict-XML close-wrong pointwise negatives.",
    )
    parser.add_argument(
        "--duplicate-exact-nonstrict",
        type=int,
        default=3,
        help="Extra copies for exact-but-non-strict-XML pointwise rows.",
    )
    return parser


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


def normalize_question(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def row_quality(row: dict[str, Any]) -> float:
    return float(row.get("quality_score", 0.0))


def pointwise_augmented_rows(
    base_rows: list[dict[str, Any]],
    duplicate_close_wrong: int,
    duplicate_exact_nonstrict: int,
    close_error_threshold: int,
) -> list[dict[str, Any]]:
    rows = list(base_rows)
    for row in base_rows:
        exact = bool(row.get("exact_match"))
        strict = bool(row.get("has_strict_xml"))
        numeric = bool(row.get("is_numeric_answer"))
        abs_error = row.get("abs_error")

        if exact and numeric and not strict:
            for copy_index in range(duplicate_exact_nonstrict):
                copied = dict(row)
                copied["augmentation_tag"] = f"exact_non_strict_dup_{copy_index + 1}"
                rows.append(copied)

        if (
            not exact
            and strict
            and numeric
            and abs_error is not None
            and int(abs_error) <= close_error_threshold
        ):
            for copy_index in range(duplicate_close_wrong):
                copied = dict(row)
                copied["augmentation_tag"] = f"strict_close_wrong_dup_{copy_index + 1}"
                rows.append(copied)
    return rows


def build_pairwise_row(
    question: str,
    gold_answer: str,
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    pair_type: str,
) -> dict[str, Any]:
    chosen_quality = row_quality(chosen)
    rejected_quality = row_quality(rejected)
    return {
        "question": question,
        "gold_answer": gold_answer,
        "chosen": chosen["completion"],
        "rejected": rejected["completion"],
        "chosen_answer": chosen.get("predicted_answer", ""),
        "rejected_answer": rejected.get("predicted_answer", ""),
        "chosen_source": chosen.get("source_run", ""),
        "rejected_source": rejected.get("source_run", ""),
        "chosen_exact_match": bool(chosen.get("exact_match")),
        "rejected_exact_match": bool(rejected.get("exact_match")),
        "chosen_abs_error": chosen.get("abs_error"),
        "rejected_abs_error": rejected.get("abs_error"),
        "preference_margin": chosen_quality - rejected_quality,
        "pair_type": pair_type,
    }


def build_augmented_pairwise_rows(
    base_pointwise_rows: list[dict[str, Any]],
    base_pairwise_rows: list[dict[str, Any]],
    close_error_threshold: int,
    medium_error_threshold: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_pointwise_rows:
        grouped[normalize_question(row.get("question"))].append(row)

    augmented_rows = list(base_pairwise_rows)
    for question, rows in grouped.items():
        gold_answer = str(rows[0].get("gold_answer", ""))
        strict_exact = sorted(
            [
                row
                for row in rows
                if bool(row.get("exact_match"))
                and bool(row.get("has_strict_xml"))
                and bool(row.get("is_numeric_answer"))
            ],
            key=row_quality,
            reverse=True,
        )
        if not strict_exact:
            continue

        chosen = strict_exact[0]
        exact_non_strict = sorted(
            [
                row
                for row in rows
                if bool(row.get("exact_match"))
                and bool(row.get("is_numeric_answer"))
                and not bool(row.get("has_strict_xml"))
            ],
            key=row_quality,
            reverse=True,
        )
        close_wrong = sorted(
            [
                row
                for row in rows
                if not bool(row.get("exact_match"))
                and bool(row.get("has_strict_xml"))
                and bool(row.get("is_numeric_answer"))
                and row.get("abs_error") is not None
                and int(row["abs_error"]) <= close_error_threshold
            ],
            key=row_quality,
            reverse=True,
        )
        medium_wrong = sorted(
            [
                row
                for row in rows
                if not bool(row.get("exact_match"))
                and bool(row.get("has_strict_xml"))
                and bool(row.get("is_numeric_answer"))
                and row.get("abs_error") is not None
                and close_error_threshold < int(row["abs_error"]) <= medium_error_threshold
            ],
            key=row_quality,
            reverse=True,
        )

        if exact_non_strict:
            augmented_rows.append(
                build_pairwise_row(
                    question=question,
                    gold_answer=gold_answer,
                    chosen=chosen,
                    rejected=exact_non_strict[0],
                    pair_type="strict_exact_vs_exact_non_strict",
                )
            )
        if close_wrong:
            augmented_rows.append(
                build_pairwise_row(
                    question=question,
                    gold_answer=gold_answer,
                    chosen=chosen,
                    rejected=close_wrong[0],
                    pair_type="strict_exact_vs_close_wrong",
                )
            )
        if medium_wrong:
            augmented_rows.append(
                build_pairwise_row(
                    question=question,
                    gold_answer=gold_answer,
                    chosen=chosen,
                    rejected=medium_wrong[0],
                    pair_type="strict_exact_vs_medium_wrong",
                )
            )
    return augmented_rows


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pointwise_in = input_dir / "pointwise_verifier_train.jsonl"
    pairwise_in = input_dir / "pairwise_verifier_train.jsonl"
    pointwise_rows = load_jsonl(pointwise_in)
    pairwise_rows = load_jsonl(pairwise_in)

    pointwise_out_rows = pointwise_augmented_rows(
        base_rows=pointwise_rows,
        duplicate_close_wrong=args.duplicate_close_wrong,
        duplicate_exact_nonstrict=args.duplicate_exact_nonstrict,
        close_error_threshold=args.close_error_threshold,
    )
    pairwise_out_rows = build_augmented_pairwise_rows(
        base_pointwise_rows=pointwise_rows,
        base_pairwise_rows=pairwise_rows,
        close_error_threshold=args.close_error_threshold,
        medium_error_threshold=args.medium_error_threshold,
    )

    pointwise_out = output_dir / "pointwise_verifier_train.jsonl"
    pairwise_out = output_dir / "pairwise_verifier_train.jsonl"
    write_jsonl(pointwise_out, pointwise_out_rows)
    write_jsonl(pairwise_out, pairwise_out_rows)

    report = {
        "input_pointwise_rows": len(pointwise_rows),
        "output_pointwise_rows": len(pointwise_out_rows),
        "input_pairwise_rows": len(pairwise_rows),
        "output_pairwise_rows": len(pairwise_out_rows),
        "pair_type_counts": dict(Counter(row.get("pair_type", "?") for row in pairwise_out_rows)),
        "augmentation_tags": dict(
            Counter(
                row.get("augmentation_tag", "base")
                for row in pointwise_out_rows
            )
        ),
        "config": {
            "close_error_threshold": args.close_error_threshold,
            "medium_error_threshold": args.medium_error_threshold,
            "duplicate_close_wrong": args.duplicate_close_wrong,
            "duplicate_exact_nonstrict": args.duplicate_exact_nonstrict,
        },
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
