from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import FeatureUnion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a lightweight verifier baseline from pointwise/pairwise GSM8K data."
    )
    parser.add_argument(
        "--pointwise-path",
        default="/home/user/图片/gsm8k_improved/verifier_dataset_v1/pointwise_verifier_train.jsonl",
        help="Path to pointwise verifier dataset.",
    )
    parser.add_argument(
        "--pairwise-path",
        default="/home/user/图片/gsm8k_improved/verifier_dataset_v1/pairwise_verifier_train.jsonl",
        help="Path to pairwise verifier dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/user/图片/gsm8k_improved/verifier_model_v1",
        help="Directory for model bundle and metrics.",
    )
    parser.add_argument(
        "--dev-mod",
        type=int,
        default=5,
        help="Use question hash modulo dev-mod to build a deterministic held-out split.",
    )
    parser.add_argument(
        "--dev-bucket",
        type=int,
        default=0,
        help="Which modulo bucket becomes the held-out split.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50000,
        help="Maximum TF-IDF feature count.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for TF-IDF terms.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=4.0,
        help="Inverse regularization strength for logistic regression.",
    )
    parser.add_argument(
        "--training-mode",
        choices=["pointwise", "pairwise"],
        default="pointwise",
        help="Whether to train a pointwise classifier or a direct pairwise ranker.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["word", "char", "hybrid"],
        default="word",
        help="Text feature family for the verifier.",
    )
    parser.add_argument(
        "--strict-positive-only",
        action="store_true",
        help="Relabel pointwise positives to require exact match plus strict XML plus numeric answer.",
    )
    parser.add_argument(
        "--use-quality-weights",
        action="store_true",
        help="Weight pointwise training rows by normalized quality score margin.",
    )
    parser.add_argument(
        "--include-structured-features",
        action="store_true",
        help="Append explicit arithmetic/format metadata features to the text representation.",
    )
    return parser


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def question_bucket(question: str, dev_mod: int) -> int:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % dev_mod


def metadata_tokens(
    has_answer_tag: bool,
    has_strict_xml: bool,
    is_numeric_answer: bool,
    completion: str,
    predicted_answer: str,
) -> str:
    length_bucket = min(len(completion) // 80, 12)
    answer_bucket = min(len(predicted_answer) // 8, 8)
    return " ".join(
        [
            f"meta_answer_tag_{int(has_answer_tag)}",
            f"meta_strict_xml_{int(has_strict_xml)}",
            f"meta_numeric_{int(is_numeric_answer)}",
            f"meta_completion_bucket_{length_bucket}",
            f"meta_answer_len_bucket_{answer_bucket}",
        ]
    )


def quality_bucket_tokens(quality_score: float) -> str:
    clipped = max(-50.0, min(1050.0, float(quality_score)))
    coarse = int(clipped // 50)
    fine = int(clipped // 10)
    return f"meta_quality_bucket_{coarse} meta_quality_fine_{fine}"


def completion_structure_tokens(completion: str, predicted_answer: str) -> str:
    lowered = completion.lower()
    reasoning_open = int("<reasoning>" in lowered)
    reasoning_close = int("</reasoning>" in lowered)
    answer_open = int("<answer>" in lowered)
    answer_close = int("</answer>" in lowered)
    equation_count = sum(1 for ch in completion if ch == "=")
    digit_count = sum(1 for ch in completion if ch.isdigit())
    newline_count = completion.count("\n")
    dollar_count = completion.count("$")
    predicted_sign = "neg" if predicted_answer.startswith("-") else "nonneg"
    return " ".join(
        [
            f"meta_reasoning_open_{reasoning_open}",
            f"meta_reasoning_close_{reasoning_close}",
            f"meta_answer_open_{answer_open}",
            f"meta_answer_close_{answer_close}",
            f"meta_equation_bucket_{min(equation_count, 8)}",
            f"meta_digit_bucket_{min(digit_count, 16)}",
            f"meta_newline_bucket_{min(newline_count, 12)}",
            f"meta_dollar_bucket_{min(dollar_count, 8)}",
            f"meta_predicted_sign_{predicted_sign}",
        ]
    )


def pointwise_label(row: dict[str, Any], strict_positive_only: bool) -> int:
    if not strict_positive_only:
        return int(row["label"])
    return int(
        bool(row.get("exact_match"))
        and bool(row.get("has_strict_xml"))
        and bool(row.get("is_numeric_answer"))
    )


def pointwise_sample_weight(row: dict[str, Any], strict_positive_only: bool, use_quality_weights: bool) -> float:
    label = pointwise_label(row, strict_positive_only)
    if not use_quality_weights:
        return 1.0
    quality_score = float(row.get("quality_score", 0.0))
    if label == 1:
        return 1.0 + min(4.0, max(0.0, quality_score - 1000.0) / 20.0)
    return 1.0 + min(3.0, max(0.0, 50.0 - quality_score) / 20.0)


def build_pointwise_text(row: dict[str, Any], include_structured_features: bool = False) -> str:
    question = normalize_text(row.get("question"))
    completion = normalize_text(row.get("completion"))
    predicted_answer = normalize_text(row.get("predicted_answer"))
    meta = metadata_tokens(
        bool(row.get("has_answer_tag")),
        bool(row.get("has_strict_xml")),
        bool(row.get("is_numeric_answer")),
        completion,
        predicted_answer,
    )
    lines = [
        f"question: {question}",
        f"predicted_answer: {predicted_answer}",
        f"completion: {completion}",
        meta,
    ]
    if include_structured_features:
        lines.append(completion_structure_tokens(completion, predicted_answer))
        lines.append(quality_bucket_tokens(float(row.get("quality_score", 0.0))))
    return "\n".join(lines)


def build_pair_completion_text(
    question: str,
    completion: str,
    predicted_answer: str,
    include_structured_features: bool = False,
) -> str:
    question = normalize_text(question)
    completion = normalize_text(completion)
    predicted_answer = normalize_text(predicted_answer)
    answer_tag = "<answer>" in completion.lower()
    strict_xml = "<reasoning>" in completion.lower() and "</answer>" in completion.lower()
    numeric_answer = any(ch.isdigit() for ch in predicted_answer)
    meta = metadata_tokens(answer_tag, strict_xml, numeric_answer, completion, predicted_answer)
    lines = [
        f"question: {question}",
        f"predicted_answer: {predicted_answer}",
        f"completion: {completion}",
        meta,
    ]
    if include_structured_features:
        lines.append(completion_structure_tokens(completion, predicted_answer))
    return "\n".join(lines)


def split_rows_by_question(
    rows: list[dict[str, Any]],
    question_key: str,
    dev_mod: int,
    dev_bucket: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    for row in rows:
        question = normalize_text(row.get(question_key))
        if question_bucket(question, dev_mod) == dev_bucket:
            dev_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, dev_rows


def safe_roc_auc(y_true: list[int], y_score: list[float]) -> float | None:
    labels = set(y_true)
    if len(labels) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def build_vectorizer(feature_mode: str, max_features: int, min_df: int):
    if feature_mode == "word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=max_features,
            lowercase=True,
            sublinear_tf=True,
        )
    if feature_mode == "char":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=min_df,
            max_features=max_features,
            lowercase=True,
            sublinear_tf=True,
        )
    word_features = max(1000, max_features // 2)
    char_features = max(1000, max_features - word_features)
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_features=word_features,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=min_df,
                    max_features=char_features,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def main() -> int:
    args = build_parser().parse_args()

    pointwise_path = Path(args.pointwise_path).expanduser()
    pairwise_path = Path(args.pairwise_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pointwise_rows = load_jsonl(pointwise_path)
    pairwise_rows = load_jsonl(pairwise_path)

    pointwise_train, pointwise_dev = split_rows_by_question(
        pointwise_rows,
        question_key="question",
        dev_mod=args.dev_mod,
        dev_bucket=args.dev_bucket,
    )
    pairwise_train, pairwise_dev = split_rows_by_question(
        pairwise_rows,
        question_key="question",
        dev_mod=args.dev_mod,
        dev_bucket=args.dev_bucket,
    )

    vectorizer = build_vectorizer(args.feature_mode, args.max_features, args.min_df)

    model = LogisticRegression(
        C=args.c,
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
    )

    pointwise_metrics: dict[str, Any] = {}
    pairwise_metrics: dict[str, Any] = {}

    if args.training_mode == "pointwise":
        train_texts = [build_pointwise_text(row, args.include_structured_features) for row in pointwise_train]
        train_labels = [pointwise_label(row, args.strict_positive_only) for row in pointwise_train]
        train_weights = [
            pointwise_sample_weight(row, args.strict_positive_only, args.use_quality_weights)
            for row in pointwise_train
        ]
        dev_texts = [build_pointwise_text(row, args.include_structured_features) for row in pointwise_dev]
        dev_labels = [pointwise_label(row, args.strict_positive_only) for row in pointwise_dev]

        x_train = vectorizer.fit_transform(train_texts)
        x_dev = vectorizer.transform(dev_texts)
        model.fit(x_train, train_labels, sample_weight=train_weights)

        dev_probs = model.predict_proba(x_dev)[:, 1]
        dev_preds = (dev_probs >= 0.5).astype(int)

        pointwise_metrics = {
            "train_rows": len(pointwise_train),
            "dev_rows": len(pointwise_dev),
            "train_positive_rate": sum(train_labels) / len(train_labels) if train_labels else 0.0,
            "dev_positive_rate": sum(dev_labels) / len(dev_labels) if dev_labels else 0.0,
            "dev_accuracy": float(accuracy_score(dev_labels, dev_preds)) if dev_labels else 0.0,
            "dev_average_precision": float(average_precision_score(dev_labels, dev_probs)) if dev_labels else 0.0,
            "dev_roc_auc": safe_roc_auc(dev_labels, list(dev_probs)),
        }

        pairwise_correct = 0
        pairwise_total = 0
        pairwise_ties = 0
        for row in pairwise_dev:
            chosen_text = build_pair_completion_text(
                question=row["question"],
                completion=row["chosen"],
                predicted_answer=row.get("chosen_answer", ""),
                include_structured_features=args.include_structured_features,
            )
            rejected_text = build_pair_completion_text(
                question=row["question"],
                completion=row["rejected"],
                predicted_answer=row.get("rejected_answer", ""),
                include_structured_features=args.include_structured_features,
            )
            chosen_score = float(model.predict_proba(vectorizer.transform([chosen_text]))[0, 1])
            rejected_score = float(model.predict_proba(vectorizer.transform([rejected_text]))[0, 1])
            if chosen_score == rejected_score:
                pairwise_ties += 1
            if chosen_score >= rejected_score:
                pairwise_correct += 1
            pairwise_total += 1

        pairwise_metrics = {
            "train_rows": len(pairwise_train),
            "dev_rows": len(pairwise_dev),
            "dev_pairwise_accuracy": (pairwise_correct / pairwise_total) if pairwise_total else 0.0,
            "dev_pairwise_ties": pairwise_ties,
        }
    else:
        pair_train_texts: list[str] = []
        for row in pairwise_train:
            pair_train_texts.append(
                build_pair_completion_text(row["question"], row["chosen"], row.get("chosen_answer", ""))
                if not args.include_structured_features
                else build_pair_completion_text(
                    row["question"], row["chosen"], row.get("chosen_answer", ""), include_structured_features=True
                )
            )
            pair_train_texts.append(
                build_pair_completion_text(row["question"], row["rejected"], row.get("rejected_answer", ""))
                if not args.include_structured_features
                else build_pair_completion_text(
                    row["question"], row["rejected"], row.get("rejected_answer", ""), include_structured_features=True
                )
            )
        vectorizer.fit(pair_train_texts)

        chosen_train = vectorizer.transform(
            [
                build_pair_completion_text(
                    row["question"],
                    row["chosen"],
                    row.get("chosen_answer", ""),
                    include_structured_features=args.include_structured_features,
                )
                for row in pairwise_train
            ]
        )
        rejected_train = vectorizer.transform(
            [
                build_pair_completion_text(
                    row["question"],
                    row["rejected"],
                    row.get("rejected_answer", ""),
                    include_structured_features=args.include_structured_features,
                )
                for row in pairwise_train
            ]
        )
        chosen_dev = vectorizer.transform(
            [
                build_pair_completion_text(
                    row["question"],
                    row["chosen"],
                    row.get("chosen_answer", ""),
                    include_structured_features=args.include_structured_features,
                )
                for row in pairwise_dev
            ]
        )
        rejected_dev = vectorizer.transform(
            [
                build_pair_completion_text(
                    row["question"],
                    row["rejected"],
                    row.get("rejected_answer", ""),
                    include_structured_features=args.include_structured_features,
                )
                for row in pairwise_dev
            ]
        )

        x_train = vstack([chosen_train - rejected_train, rejected_train - chosen_train])
        y_train = [1] * len(pairwise_train) + [0] * len(pairwise_train)
        x_dev = vstack([chosen_dev - rejected_dev, rejected_dev - chosen_dev])
        y_dev = [1] * len(pairwise_dev) + [0] * len(pairwise_dev)
        model.fit(x_train, y_train)

        dev_probs = model.predict_proba(x_dev)[:, 1]
        dev_preds = (dev_probs >= 0.5).astype(int)
        pairwise_forward_probs = dev_probs[: len(pairwise_dev)]
        pairwise_forward_preds = dev_preds[: len(pairwise_dev)]

        pairwise_metrics = {
            "train_rows": len(pairwise_train),
            "dev_rows": len(pairwise_dev),
            "dev_pairwise_accuracy": float(accuracy_score([1] * len(pairwise_dev), pairwise_forward_preds))
            if pairwise_dev
            else 0.0,
            "dev_pairwise_average_precision": float(average_precision_score(y_dev, dev_probs)) if y_dev else 0.0,
            "dev_pairwise_roc_auc": safe_roc_auc(y_dev, list(dev_probs)),
            "dev_pairwise_margin_mean": float(sum(abs(prob - 0.5) for prob in pairwise_forward_probs) / len(pairwise_forward_probs))
            if pairwise_forward_probs.size
            else 0.0,
        }

    metrics = {
        "split": {
            "dev_mod": args.dev_mod,
            "dev_bucket": args.dev_bucket,
        },
        "pointwise": pointwise_metrics,
        "pairwise_eval": pairwise_metrics,
        "model": {
            "type": f"tfidf_logistic_regression_{args.training_mode}",
            "solver": "liblinear",
            "c": args.c,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "training_mode": args.training_mode,
            "feature_mode": args.feature_mode,
            "strict_positive_only": args.strict_positive_only,
            "use_quality_weights": args.use_quality_weights,
            "include_structured_features": args.include_structured_features,
        },
    }

    bundle_path = output_dir / "verifier_bundle.pkl"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.txt"

    with bundle_path.open("wb") as fh:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "model": model,
                "metrics": metrics,
                "pointwise_path": str(pointwise_path),
                "pairwise_path": str(pairwise_path),
            },
            fh,
        )

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report = (
        "Verifier baseline training\n"
        "==========================\n"
        f"training_mode: {args.training_mode}\n"
        f"feature_mode: {args.feature_mode}\n"
        f"pointwise_train_rows: {pointwise_metrics.get('train_rows', 0)}\n"
        f"pointwise_dev_rows: {pointwise_metrics.get('dev_rows', 0)}\n"
        f"pointwise_dev_accuracy: {pointwise_metrics.get('dev_accuracy')}\n"
        f"pointwise_dev_average_precision: {pointwise_metrics.get('dev_average_precision')}\n"
        f"pointwise_dev_roc_auc: {pointwise_metrics.get('dev_roc_auc')}\n"
        f"pairwise_train_rows: {pairwise_metrics['train_rows']}\n"
        f"pairwise_dev_rows: {pairwise_metrics['dev_rows']}\n"
        f"pairwise_dev_accuracy: {pairwise_metrics['dev_pairwise_accuracy']:.6f}\n"
        f"pairwise_dev_average_precision: {pairwise_metrics.get('dev_pairwise_average_precision')}\n"
        f"pairwise_dev_roc_auc: {pairwise_metrics.get('dev_pairwise_roc_auc')}\n"
        f"pairwise_dev_margin_mean: {pairwise_metrics.get('dev_pairwise_margin_mean')}\n"
        f"pairwise_dev_ties: {pairwise_metrics.get('dev_pairwise_ties')}\n"
        f"bundle_path: {bundle_path}\n"
        f"metrics_path: {metrics_path}\n"
    )
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
