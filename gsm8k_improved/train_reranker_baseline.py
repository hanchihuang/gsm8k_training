from __future__ import annotations

import argparse
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
        description="Train a lightweight reranker from candidate-level pairwise/listwise data."
    )
    parser.add_argument(
        "--pairwise-path",
        default="/home/user/图片/gsm8k_improved/reranker_dataset_from_eval200_20260404/pairwise_reranker_train.jsonl",
        help="Path to pairwise reranker dataset.",
    )
    parser.add_argument(
        "--listwise-path",
        default="/home/user/图片/gsm8k_improved/reranker_dataset_from_eval200_20260404/listwise_reranker_train.jsonl",
        help="Path to listwise reranker dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/user/图片/gsm8k_improved/reranker_model_v1",
        help="Directory for reranker bundle and metrics.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=60000,
        help="Maximum TF-IDF features.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=1,
        help="Minimum document frequency.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=2.0,
        help="Inverse regularization strength for logistic regression.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["word", "char", "hybrid"],
        default="hybrid",
        help="Text feature family.",
    )
    parser.add_argument(
        "--dev-count",
        type=int,
        default=40,
        help="Hold out the last N listwise rows for validation.",
    )
    parser.add_argument(
        "--training-mode",
        choices=["pairwise", "pointwise"],
        default="pointwise",
        help="Reranker training objective.",
    )
    parser.add_argument(
        "--positive-upweight",
        type=float,
        default=3.0,
        help="Extra sample weight for positive listwise candidates in pointwise mode.",
    )
    parser.add_argument(
        "--hard-negative-upweight",
        type=float,
        default=2.5,
        help="Extra sample weight for hard negatives in pointwise mode.",
    )
    parser.add_argument(
        "--quality-weight-scale",
        type=float,
        default=1.0,
        help="Scale for incorporating listwise quality_score into pointwise sample weights.",
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


def safe_roc_auc(y_true: list[int], y_score: list[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


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


def candidate_meta_text(candidate: dict[str, Any]) -> str:
    return " ".join(
        [
            f"meta_answer_tag_{int(bool(candidate.get('has_answer_tag')))}",
            f"meta_strict_xml_{int(bool(candidate.get('has_strict_xml')))}",
            f"meta_numeric_{int(bool(candidate.get('is_numeric_answer')))}",
            f"meta_conf_bucket_{int((float(candidate.get('confidence', 0.0)) + 5.0) * 5)}",
            f"meta_lowconf_bucket_{int(float(candidate.get('low_confidence_ratio', 0.0)) * 20)}",
            f"meta_consensus_{int(candidate.get('consensus_count', 0) or 0)}",
            f"meta_verifier_bucket_{int((float(candidate.get('verifier_score', 0.0)) + 1.0) * 10)}",
            f"meta_equation_bucket_{int(float(candidate.get('equation_support', 0.0)) * 20)}",
            f"meta_novelty_bucket_{int(float(candidate.get('novelty', 0.0)) * 20)}",
            f"meta_trunc_bucket_{int(float(candidate.get('truncated_xml_penalty', 0.0)) * 20)}",
        ]
    )


def build_candidate_text(question: str, candidate: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"question: {normalize_text(question)}",
            f"predicted_answer: {normalize_text(candidate.get('predicted_answer', ''))}",
            f"completion: {normalize_text(candidate.get('text', ''))}",
            candidate_meta_text(candidate),
        ]
    )


def build_candidate_from_pair(text: str, answer: str, rerank_score: Any, verifier_score: Any) -> dict[str, Any]:
    return {
        "text": text,
        "predicted_answer": answer,
        "rerank_score": float(rerank_score or 0.0),
        "verifier_score": float(verifier_score or 0.0),
        "has_answer_tag": "<answer>" in str(text).lower(),
        "has_strict_xml": ("<reasoning>" in str(text).lower()) and ("</answer>" in str(text).lower()),
        "is_numeric_answer": normalize_text(answer).lstrip("-").isdigit(),
        "confidence": 0.0,
        "low_confidence_ratio": 0.0,
        "consensus_count": 1,
        "equation_support": 0.0,
        "novelty": 0.0,
        "truncated_xml_penalty": 0.0,
    }


def bounded_quality_bonus(candidate: dict[str, Any], scale: float) -> float:
    quality_score = float(candidate.get("quality_score", 0.0) or 0.0)
    if quality_score <= 0.0 or scale <= 0.0:
        return 0.0
    return min(quality_score / 1000.0, 1.5) * scale


def build_pointwise_training_rows(
    listwise_rows: list[dict[str, Any]],
    positive_upweight: float,
    hard_negative_upweight: float,
    quality_weight_scale: float,
) -> tuple[list[str], list[int], list[float]]:
    texts: list[str] = []
    labels: list[int] = []
    weights: list[float] = []

    for row in listwise_rows:
        candidates = row.get("candidates", [])
        positive_candidates = [candidate for candidate in candidates if int(candidate.get("label", 0)) == 1]
        positive_rerank_floor = max(
            (float(candidate.get("rerank_score", 0.0) or 0.0) for candidate in positive_candidates),
            default=float("-inf"),
        )

        for candidate in candidates:
            label = int(candidate.get("label", 0))
            sample_weight = 1.0
            if label == 1:
                sample_weight *= positive_upweight
                sample_weight += bounded_quality_bonus(candidate, quality_weight_scale)
            elif float(candidate.get("rerank_score", 0.0) or 0.0) >= positive_rerank_floor:
                sample_weight *= hard_negative_upweight

            texts.append(build_candidate_text(row["question"], candidate))
            labels.append(label)
            weights.append(sample_weight)

    return texts, labels, weights


def main() -> int:
    args = build_parser().parse_args()
    pairwise_path = Path(args.pairwise_path).expanduser()
    listwise_path = Path(args.listwise_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise_rows = load_jsonl(pairwise_path)
    listwise_rows = load_jsonl(listwise_path)

    dev_count = min(args.dev_count, max(1, len(listwise_rows) // 5), len(listwise_rows))
    dev_questions = {normalize_text(row["question"]) for row in listwise_rows[-dev_count:]}

    pairwise_train = [row for row in pairwise_rows if normalize_text(row["question"]) not in dev_questions]
    pairwise_dev = [row for row in pairwise_rows if normalize_text(row["question"]) in dev_questions]
    listwise_train = [row for row in listwise_rows if normalize_text(row["question"]) not in dev_questions]
    listwise_dev = [row for row in listwise_rows if normalize_text(row["question"]) in dev_questions]

    vectorizer = build_vectorizer(args.feature_mode, args.max_features, args.min_df)
    model = LogisticRegression(
        C=args.c,
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
    )

    vectorizer_fit_texts: list[str] = []
    for row in listwise_train:
        for candidate in row.get("candidates", []):
            vectorizer_fit_texts.append(build_candidate_text(row["question"], candidate))
    for row in pairwise_train:
        vectorizer_fit_texts.append(
            build_candidate_text(
                row["question"],
                build_candidate_from_pair(
                    row["chosen"],
                    row.get("chosen_answer", ""),
                    row.get("chosen_rerank_score"),
                    row.get("chosen_verifier_score"),
                ),
            )
        )
        vectorizer_fit_texts.append(
            build_candidate_text(
                row["question"],
                build_candidate_from_pair(
                    row["rejected"],
                    row.get("rejected_answer", ""),
                    row.get("rejected_rerank_score"),
                    row.get("rejected_verifier_score"),
                ),
            )
        )
    vectorizer.fit(vectorizer_fit_texts)

    if args.training_mode == "pairwise":
        chosen_train = vectorizer.transform(
            [
                build_candidate_text(
                    row["question"],
                    build_candidate_from_pair(
                        row["chosen"],
                        row.get("chosen_answer", ""),
                        row.get("chosen_rerank_score"),
                        row.get("chosen_verifier_score"),
                    ),
                )
                for row in pairwise_train
            ]
        )
        rejected_train = vectorizer.transform(
            [
                build_candidate_text(
                    row["question"],
                    build_candidate_from_pair(
                        row["rejected"],
                        row.get("rejected_answer", ""),
                        row.get("rejected_rerank_score"),
                        row.get("rejected_verifier_score"),
                    ),
                )
                for row in pairwise_train
            ]
        )
        chosen_dev = vectorizer.transform(
            [
                build_candidate_text(
                    row["question"],
                    build_candidate_from_pair(
                        row["chosen"],
                        row.get("chosen_answer", ""),
                        row.get("chosen_rerank_score"),
                        row.get("chosen_verifier_score"),
                    ),
                )
                for row in pairwise_dev
            ]
        )
        rejected_dev = vectorizer.transform(
            [
                build_candidate_text(
                    row["question"],
                    build_candidate_from_pair(
                        row["rejected"],
                        row.get("rejected_answer", ""),
                        row.get("rejected_rerank_score"),
                        row.get("rejected_verifier_score"),
                    ),
                )
                for row in pairwise_dev
            ]
        )

        x_train = vstack([chosen_train - rejected_train, rejected_train - chosen_train])
        y_train = [1] * len(pairwise_train) + [0] * len(pairwise_train)
        model.fit(x_train, y_train)

        x_dev = vstack([chosen_dev - rejected_dev, rejected_dev - chosen_dev]) if pairwise_dev else None
        y_dev = [1] * len(pairwise_dev) + [0] * len(pairwise_dev)
        dev_probs = model.predict_proba(x_dev)[:, 1] if pairwise_dev else []
        dev_preds = (dev_probs >= 0.5).astype(int) if pairwise_dev else []
        forward_probs = dev_probs[: len(pairwise_dev)] if pairwise_dev else []
        forward_preds = dev_preds[: len(pairwise_dev)] if pairwise_dev else []
    else:
        train_texts, y_train, sample_weights = build_pointwise_training_rows(
            listwise_train,
            positive_upweight=args.positive_upweight,
            hard_negative_upweight=args.hard_negative_upweight,
            quality_weight_scale=args.quality_weight_scale,
        )
        x_train = vectorizer.transform(train_texts)
        model.fit(x_train, y_train, sample_weight=sample_weights)

        forward_probs: list[float] = []
        forward_preds: list[int] = []
        y_dev: list[int] = []
        dev_probs: list[float] = []
        for row in pairwise_dev:
            chosen_text = build_candidate_text(
                row["question"],
                build_candidate_from_pair(
                    row["chosen"],
                    row.get("chosen_answer", ""),
                    row.get("chosen_rerank_score"),
                    row.get("chosen_verifier_score"),
                ),
            )
            rejected_text = build_candidate_text(
                row["question"],
                build_candidate_from_pair(
                    row["rejected"],
                    row.get("rejected_answer", ""),
                    row.get("rejected_rerank_score"),
                    row.get("rejected_verifier_score"),
                ),
            )
            candidate_features = vectorizer.transform([chosen_text, rejected_text])
            candidate_scores = model.predict_proba(candidate_features)[:, 1]
            forward_prob = float(candidate_scores[0] > candidate_scores[1])
            forward_probs.append(forward_prob)
            forward_preds.append(int(candidate_scores[0] > candidate_scores[1]))
            dev_probs.extend([float(candidate_scores[0]), float(candidate_scores[1])])
            y_dev.extend([1, 0])

    listwise_correct = 0
    listwise_total = 0
    for row in listwise_dev:
        scored = []
        for candidate in row["candidates"]:
            text = build_candidate_text(row["question"], candidate)
            score = float(model.decision_function(vectorizer.transform([text]))[0])
            scored.append((score, candidate))
        if not scored:
            continue
        best = max(scored, key=lambda item: item[0])[1]
        if int(best.get("label", 0)) == 1:
            listwise_correct += 1
        listwise_total += 1

    metrics = {
        "pairwise": {
            "train_rows": len(pairwise_train),
            "dev_rows": len(pairwise_dev),
            "dev_pairwise_accuracy": float(accuracy_score([1] * len(pairwise_dev), forward_preds))
            if pairwise_dev
            else 0.0,
            "dev_pairwise_average_precision": float(average_precision_score(y_dev, dev_probs)) if pairwise_dev else 0.0,
            "dev_pairwise_roc_auc": safe_roc_auc(y_dev, list(dev_probs)) if pairwise_dev else None,
        },
        "listwise": {
            "dev_rows": listwise_total,
            "dev_top1_accuracy": (listwise_correct / listwise_total) if listwise_total else 0.0,
        },
        "model": {
            "type": "tfidf_logistic_regression_pairwise_reranker",
            "solver": "liblinear",
            "c": args.c,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "feature_mode": args.feature_mode,
            "dev_count": dev_count,
            "training_mode": args.training_mode,
            "positive_upweight": args.positive_upweight,
            "hard_negative_upweight": args.hard_negative_upweight,
            "quality_weight_scale": args.quality_weight_scale,
        },
    }

    bundle_path = output_dir / "reranker_bundle.pkl"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.txt"
    with bundle_path.open("wb") as fh:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "model": model,
                "metrics": metrics,
                "pairwise_path": str(pairwise_path),
                "listwise_path": str(listwise_path),
            },
            fh,
        )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report = (
        "Reranker baseline training\n"
        "==========================\n"
        f"pairwise_train_rows: {metrics['pairwise']['train_rows']}\n"
        f"pairwise_dev_rows: {metrics['pairwise']['dev_rows']}\n"
        f"pairwise_dev_accuracy: {metrics['pairwise']['dev_pairwise_accuracy']}\n"
        f"pairwise_dev_average_precision: {metrics['pairwise']['dev_pairwise_average_precision']}\n"
        f"pairwise_dev_roc_auc: {metrics['pairwise']['dev_pairwise_roc_auc']}\n"
        f"listwise_dev_rows: {metrics['listwise']['dev_rows']}\n"
        f"listwise_dev_top1_accuracy: {metrics['listwise']['dev_top1_accuracy']}\n"
        f"bundle_path: {bundle_path}\n"
    )
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
