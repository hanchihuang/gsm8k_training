from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from multi_agent_autoresearch.engine import AutoResearchEngine
from multi_agent_autoresearch.models import RunConfig, utc_now


def _load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'").strip('"')
    return env


def _safe_metric(run_summary: Path, section: str) -> tuple[float, int]:
    payload = json.loads(run_summary.read_text(encoding="utf-8"))
    section_payload = payload.get(section, {}) or {}
    return (
        float(section_payload.get("exact_match_rate", 0.0) or 0.0),
        int(section_payload.get("exact_match_count", 0) or 0),
    )


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _signature_from_overrides(overrides: dict[str, str]) -> str:
    if not overrides:
        return "baseline"
    return "|".join(f"{key}={overrides[key]}" for key in sorted(overrides))


def _infer_required_eval_dataset_name(env: dict[str, str]) -> str | None:
    dataset_source = env.get("DATASET_SOURCE", "").strip().lower()
    dataset_split = env.get("DATASET_SPLIT", "").strip().lower()
    explicit_eval_split = env.get("EVAL_SPLIT", "").strip().lower()
    validation_mod = int(env.get("TRAIN_VALIDATION_MOD", "0") or "0")
    validation_bucket = int(env.get("TRAIN_VALIDATION_BUCKET", "0") or "0")
    if dataset_source != "gsm8k":
        return None
    if dataset_split == "train" and validation_mod > 1:
        return (
            f"gsm8k_train_validation_mod{validation_mod}"
            f"_bucket{validation_bucket % validation_mod}"
        )
    if explicit_eval_split:
        return f"gsm8k_{explicit_eval_split}"
    if dataset_split:
        return "gsm8k_test" if dataset_split == "train" else f"gsm8k_{dataset_split}"
    return None


def _sanitize_label(text: str) -> str:
    sanitized = []
    for char in text.lower():
        if char.isalnum():
            sanitized.append(char)
        elif char in {"-", "_"}:
            sanitized.append(char)
        else:
            sanitized.append("_")
    compact = "".join(sanitized).strip("_")
    return compact or "proposal"


def _candidate_pass_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    pass1 = 0
    pass8 = 0
    total = len(rows)
    saw_candidate_records = False
    for row in rows:
        if row.get("exact_match"):
            pass1 += 1
        candidate_records = row.get("candidate_records") or []
        if candidate_records:
            saw_candidate_records = True
        candidate_hit = any(item.get("is_correct") for item in candidate_records)
        if candidate_hit:
            pass8 += 1
    if total and not saw_candidate_records:
        pass8 = pass1
    rate1 = pass1 / total if total else 0.0
    rate8 = pass8 / total if total else 0.0
    return {
        "pass1": rate1,
        "pass8": rate8,
        "gap": rate8 - rate1,
    }


def _slice_bottom_list(slice_metrics: dict[str, Any], limit: int = 3) -> list[str]:
    ranked = []
    for name, payload in slice_metrics.items():
        rate = float((payload or {}).get("exact_match_rate", 0.0) or 0.0)
        ranked.append((rate, name))
    ranked.sort()
    return [name for _, name in ranked[:limit]]


def _top_counts(payload: dict[str, Any], prefix: str, limit: int = 4) -> list[str]:
    ranked: list[tuple[int, str]] = []
    for name, raw_value in payload.items():
        if not name.startswith(prefix):
            continue
        ranked.append((int(raw_value or 0), name))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in ranked[:limit]]


def _scan_experiment_root(
    root: Path | None,
    metric_section: str,
    *,
    required_model_name: str | None = None,
    required_eval_dataset_name: str | None = None,
) -> dict[str, Any]:
    if root is None or not root.exists():
        return {}
    completed_runs = 0
    early_stops = 0
    best_metric = 0.0
    best_dir = ""
    top_runs: list[dict[str, Any]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        summary_path = child / "run_summary.json"
        early_stop_path = child / "early_stop.txt"
        if summary_path.exists():
            payload = _json_load(summary_path)
            model_name = str(payload.get("model_name", "") or "")
            eval_dataset_name = str(payload.get("eval_dataset_name", "") or "")
            if required_model_name and model_name and model_name != required_model_name:
                continue
            if (
                required_eval_dataset_name
                and eval_dataset_name
                and eval_dataset_name != required_eval_dataset_name
            ):
                continue
            completed_runs += 1
            section_payload = payload.get(metric_section, {}) or {}
            metric = float(section_payload.get("exact_match_rate", 0.0) or 0.0)
            if metric > best_metric:
                best_metric = metric
                best_dir = str(child)
            top_runs.append(
                {
                    "dir": str(child),
                    "metric": metric,
                    "model_name": model_name,
                    "eval_dataset_name": eval_dataset_name,
                }
            )
        elif early_stop_path.exists():
            early_stops += 1
    top_runs.sort(key=lambda item: item["metric"], reverse=True)
    return {
        "root": str(root),
        "completed_runs": completed_runs,
        "early_stops": early_stops,
        "best_metric": best_metric,
        "best_dir": best_dir,
        "top_runs": top_runs[:5],
    }


@dataclass(slots=True)
class LoopIteration:
    index: int
    label: str
    started_at: str
    completed_at: str | None = None
    base_metric: float = 0.0
    metric: float = 0.0
    exact_match_count: int = 0
    status: str = "running"
    output_dir: str = ""
    overrides: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    family: str = "baseline"
    signature: str = ""
    rationale: str = ""
    diagnosis: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GSM8KLoopConfig:
    query: str
    output_dir: Path
    baseline_env_path: Path
    script_path: Path
    local_roots: list[str] = field(default_factory=list)
    metric_section: str = "eval_before"
    max_rounds: int = 0
    sync_script: str = "/home/user/图片/gsm8k_improved/sync_experiment_to_git_repo.sh"
    sync_repo: str = "/home/user/图片/gsm8k_training_repo"
    runner_path: Path | None = None
    experiment_root: Path | None = None
    resume: bool = True
    enable_research_wave: bool = True


@dataclass(slots=True)
class GSM8KLoopState:
    config: GSM8KLoopConfig
    started_at: str
    best_metric: float
    best_exact_match_count: int
    best_label: str
    best_output_dir: str
    current_env: dict[str, str]
    iterations: list[LoopIteration] = field(default_factory=list)
    latest_diagnosis: dict[str, Any] = field(default_factory=dict)
    next_proposals: list[dict[str, Any]] = field(default_factory=list)
    external_history: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config"]["output_dir"] = str(self.config.output_dir)
        payload["config"]["baseline_env_path"] = str(self.config.baseline_env_path)
        payload["config"]["script_path"] = str(self.config.script_path)
        payload["config"]["runner_path"] = (
            str(self.config.runner_path) if self.config.runner_path is not None else None
        )
        payload["config"]["experiment_root"] = (
            str(self.config.experiment_root) if self.config.experiment_root is not None else None
        )
        return payload


@dataclass(slots=True)
class HypothesisProposal:
    label: str
    family: str
    overrides: dict[str, str]
    rationale: str
    priority: float = 0.0

    @property
    def signature(self) -> str:
        return _signature_from_overrides(self.overrides)


def _proposal_library() -> list[HypothesisProposal]:
    return [
        HypothesisProposal(
            label="selector_margin_loosened",
            family="selector",
            overrides={
                "ANSWER_AGG_MARGIN": "0.20",
                "ANSWER_AGG_PAIR_COUNT_WEIGHT": "0.5",
                "SAVE_RERANK_CANDIDATES": "1",
            },
            rationale="Loosen answer aggregation margin slightly while increasing pair-count support.",
        ),
        HypothesisProposal(
            label="selector_numc12_top_p095",
            family="selector",
            overrides={
                "EVAL_NUM_CANDIDATES": "12",
                "EVAL_RERANK_TOP_P": "0.95",
                "SAVE_RERANK_CANDIDATES": "1",
            },
            rationale="Increase candidate coverage when pass@8 is above pass@1.",
        ),
        HypothesisProposal(
            label="selector_numc12_temp07_top_p095",
            family="selector",
            overrides={
                "EVAL_NUM_CANDIDATES": "12",
                "EVAL_RERANK_TEMPERATURE": "0.7",
                "EVAL_RERANK_TOP_P": "0.95",
                "SAVE_RERANK_CANDIDATES": "1",
            },
            rationale="Broaden candidate pool diversity while retaining the 0.565 rerank stack.",
        ),
        HypothesisProposal(
            label="selector_numc12_verifier025",
            family="selector",
            overrides={
                "EVAL_NUM_CANDIDATES": "12",
                "VERIFIER_SCORE_WEIGHT": "0.25",
                "SAVE_RERANK_CANDIDATES": "1",
            },
            rationale="Lean harder on verifier score when selector instability dominates.",
        ),
        HypothesisProposal(
            label="selector_expand_profiles",
            family="selector",
            overrides={
                "EVAL_EXPAND_CANDIDATES": "1",
                "EVAL_EXPAND_CANDIDATES_EXTRA": "8",
                "EVAL_EXPAND_MULTI_PROFILE": "1",
                "EVAL_EXPAND_MIN_UNIQUE_ANSWERS": "4",
                "EVAL_EXPAND_MAX_TOP_CONSENSUS": "2",
                "SAVE_RERANK_CANDIDATES": "1",
            },
            rationale="Use profile expansion when the pool has answers but lacks diversity on weak slices.",
        ),
        HypothesisProposal(
            label="data_quality_balanced_075_train",
            family="data_quality_train",
            overrides={
                "EVAL_ONLY": "0",
                "TRAINING_METHOD": "grpo",
                "EARLY_STOP_ENABLE": "0",
                "ENABLE_DATA_QUALITY_FILTER": "1",
                "DATA_QUALITY_KEEP_RATIO": "0.75",
                "DATA_QUALITY_POLICY": "balanced",
            },
            rationale="Use moderate data-quality filtering near the successful P15 regime instead of strict pruning.",
        ),
        HypothesisProposal(
            label="data_quality_balanced_07_min16_train",
            family="data_quality_train",
            overrides={
                "EVAL_ONLY": "0",
                "TRAINING_METHOD": "grpo",
                "EARLY_STOP_ENABLE": "0",
                "ENABLE_DATA_QUALITY_FILTER": "1",
                "DATA_QUALITY_KEEP_RATIO": "0.7",
                "DATA_QUALITY_POLICY": "balanced",
                "DATA_QUALITY_MIN_SCORE": "1.6",
            },
            rationale="Stay close to the 0.56 line but add a mild cutoff to remove only the lowest-value training questions.",
        ),
        HypothesisProposal(
            label="data_quality_strict_065_min22_validation",
            family="data_quality_train",
            overrides={
                "EVAL_ONLY": "0",
                "TRAINING_METHOD": "grpo",
                "EARLY_STOP_ENABLE": "0",
                "ENABLE_DATA_QUALITY_FILTER": "1",
                "DATA_QUALITY_KEEP_RATIO": "0.65",
                "DATA_QUALITY_POLICY": "strict",
                "DATA_QUALITY_MIN_SCORE": "2.2",
            },
            rationale="Tighten training questions when validation errors cluster on percentage, relation, and remaining-vs-total modeling.",
        ),
        HypothesisProposal(
            label="prompt_answer_first",
            family="prompt",
            overrides={
                "PROMPT_TEMPLATE_MODE": "answer_first",
                "ANSWER_EXTRACTION_MODE": "xml_or_last_number",
            },
            rationale="Try answer-first prompting only if formatting remains stable but numeric extraction is weak.",
        ),
        HypothesisProposal(
            label="reward_reduced_v1_train",
            family="reward_train",
            overrides={
                "EVAL_ONLY": "0",
                "TRAINING_METHOD": "grpo",
                "EARLY_STOP_ENABLE": "0",
                "REWARD_WEIGHT_XML": "0.15",
                "REWARD_WEIGHT_NUMERIC": "0.05",
                "REWARD_WEIGHT_DISTANCE": "0.25",
                "REWARD_WEIGHT_PARTIAL": "0.0",
                "REWARD_WEIGHT_REASONING": "0.05",
                "REWARD_WEIGHT_EQUATION": "0.12",
                "REWARD_WEIGHT_BREVITY": "0.04",
                "REWARD_WEIGHT_NOVELTY": "0.06",
                "REWARD_WEIGHT_CORRECTNESS": "1.0",
            },
            rationale="Reduce wrong-high-reward pressure when reward audit says formatting dominates correctness.",
        ),
        HypothesisProposal(
            label="selector_numc12_verifier03_expand",
            family="selector",
            overrides={
                "EVAL_NUM_CANDIDATES": "12",
                "VERIFIER_SCORE_WEIGHT": "0.3",
                "EVAL_EXPAND_CANDIDATES": "1",
                "EVAL_EXPAND_CANDIDATES_EXTRA": "8",
                "SAVE_RERANK_CANDIDATES": "1",
            },
            rationale="Push verifier-assisted candidate selection harder when validation misses are dominated by rate/unit chain questions.",
        ),
        HypothesisProposal(
            label="reward_reduced_v2_train",
            family="reward_train",
            overrides={
                "EVAL_ONLY": "0",
                "TRAINING_METHOD": "grpo",
                "EARLY_STOP_ENABLE": "0",
                "REWARD_WEIGHT_XML": "0.10",
                "REWARD_WEIGHT_NUMERIC": "0.02",
                "REWARD_WEIGHT_DISTANCE": "0.20",
                "REWARD_WEIGHT_PARTIAL": "0.0",
                "REWARD_WEIGHT_REASONING": "0.02",
                "REWARD_WEIGHT_EQUATION": "0.16",
                "REWARD_WEIGHT_BREVITY": "0.06",
                "REWARD_WEIGHT_NOVELTY": "0.08",
                "REWARD_WEIGHT_CORRECTNESS": "1.25",
            },
            rationale="Push reward harder toward exact correctness after reward-down v1 if the issue persists.",
        ),
    ]


def _diagnose_run_summary(run_summary: Path, metric_section: str) -> dict[str, Any]:
    payload = _json_load(run_summary)
    section = payload.get(metric_section, {}) or {}
    rows = section.get("rows") or []
    reasoning_pattern_metrics = section.get("reasoning_pattern_metrics", {}) or {}
    error_attribution = section.get("error_attribution", {}) or {}
    diagnosis: dict[str, Any] = {
        "metric_section": metric_section,
        "eval_dataset_name": str(payload.get("eval_dataset_name", "") or ""),
        "exact_match_rate": float(section.get("exact_match_rate", 0.0) or 0.0),
        "exact_match_count": int(section.get("exact_match_count", 0) or 0),
        "answer_tag_rate": float(section.get("answer_tag_rate", 0.0) or 0.0),
        "strict_xml_rate": float(section.get("strict_xml_rate", 0.0) or 0.0),
        "numeric_answer_rate": float(section.get("numeric_answer_rate", 0.0) or 0.0),
        "correctness_reward_mean": float(section.get("correctness_reward_mean", 0.0) or 0.0),
        "distance_reward_mean": float(section.get("distance_reward_mean", 0.0) or 0.0),
        "mean_abs_error": float(section.get("mean_abs_error", 0.0) or 0.0),
        "bottom_slices": _slice_bottom_list(section.get("slice_metrics", {}) or {}),
        "reasoning_pattern_metrics": reasoning_pattern_metrics,
        "error_attribution": error_attribution,
        "top_wrong_patterns": _top_counts(error_attribution, "wrong::"),
        "top_correct_patterns": _top_counts(error_attribution, "correct::"),
    }
    if rows:
        diagnosis.update(_candidate_pass_metrics(rows))
    else:
        diagnosis.update({"pass1": diagnosis["exact_match_rate"], "pass8": diagnosis["exact_match_rate"], "gap": 0.0})
    return diagnosis


def _score_proposal(
    proposal: HypothesisProposal,
    diagnosis: dict[str, Any],
    family_discards: dict[str, int],
    external_history: dict[str, Any],
) -> float:
    score = 0.0
    gap = float(diagnosis.get("gap", 0.0) or 0.0)
    strict_xml = float(diagnosis.get("strict_xml_rate", 0.0) or 0.0)
    numeric = float(diagnosis.get("numeric_answer_rate", 0.0) or 0.0)
    distance = float(diagnosis.get("distance_reward_mean", 0.0) or 0.0)
    correctness = float(diagnosis.get("correctness_reward_mean", 0.0) or 0.0)
    weak_slices = set(diagnosis.get("bottom_slices", []) or [])
    wrong_patterns = set(diagnosis.get("top_wrong_patterns", []) or [])
    external_best = float(external_history.get("best_metric", 0.0) or 0.0)
    top_run_dirs = " ".join(item.get("dir", "") for item in external_history.get("top_runs", []) or []).lower()
    history_supports_data_quality = "checklist_p15_data_quality_filter" in top_run_dirs
    history_supports_selector = "neartop" in top_run_dirs or "supportscore" in top_run_dirs
    percentage_or_growth_errors = any("percentage_discount_growth" in item for item in wrong_patterns)
    unit_chain_errors = any("rate_ratio_unit_chain" in item for item in wrong_patterns)
    multiplicative_errors = any("multiplicative_relation" in item for item in wrong_patterns)
    remaining_errors = any("remaining_total_transition" in item for item in wrong_patterns)

    if proposal.family == "selector":
        score += max(gap, 0.0) * 10.0
        if external_best > diagnosis.get("exact_match_rate", 0.0):
            score += 0.8
        if history_supports_selector:
            score += 0.9
        if {"difference", "basic_arithmetic"} & weak_slices:
            score += 0.75
        if unit_chain_errors:
            score += 1.0
        if remaining_errors or multiplicative_errors:
            score += 0.5
        if unit_chain_errors and "verifier" in proposal.label:
            score += 0.35
        if unit_chain_errors and "expand" in proposal.label:
            score += 0.15
        if unit_chain_errors and proposal.label == "selector_numc12_verifier03_expand":
            score += 0.2
    elif proposal.family == "prompt":
        if strict_xml >= 0.98 and numeric < 0.98:
            score += 1.5
        else:
            score -= 1.0
    elif proposal.family == "data_quality_train":
        if {"percentage", "rate_or_ratio"} & weak_slices:
            score += 1.2
        if history_supports_data_quality:
            score += 1.4
        if external_best > diagnosis.get("exact_match_rate", 0.0):
            score += 0.5
        if percentage_or_growth_errors:
            score += 1.1
        if multiplicative_errors or remaining_errors:
            score += 0.8
        if percentage_or_growth_errors and "strict" in proposal.label:
            score += 0.25
    elif proposal.family == "reward_train":
        if distance >= 0.2 and correctness < 4.0:
            score += 1.8
        if {"percentage", "rate_or_ratio"} & weak_slices:
            score += 0.6
        if strict_xml >= 0.99 and numeric >= 0.99:
            score -= 1.0
        if unit_chain_errors or percentage_or_growth_errors:
            score -= 0.4

    discard_penalty = 0.6
    if proposal.family == "reward_train":
        discard_penalty = 1.2
    score -= family_discards.get(proposal.family, 0) * discard_penalty
    return score


def _build_dynamic_proposals(state: GSM8KLoopState) -> list[HypothesisProposal]:
    attempted = {iteration.signature for iteration in state.iterations if iteration.signature}
    family_discards: dict[str, int] = {}
    for iteration in state.iterations:
        if iteration.status in {"discard", "failed", "early_stop"}:
            family_discards[iteration.family] = family_discards.get(iteration.family, 0) + 1

    diagnosis = state.latest_diagnosis or {}
    proposals: list[HypothesisProposal] = []
    for proposal in _proposal_library():
        if proposal.signature in attempted:
            continue
        scored = HypothesisProposal(
            label=proposal.label,
            family=proposal.family,
            overrides=dict(proposal.overrides),
            rationale=proposal.rationale,
            priority=_score_proposal(proposal, diagnosis, family_discards, state.external_history),
        )
        proposals.append(scored)
    proposals.sort(key=lambda item: item.priority, reverse=True)
    return proposals


class GSM8KLoopRunner:
    def __init__(self, config: GSM8KLoopConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.research_dir = self.output_dir / "research"
        self.runs_dir = self.output_dir / "runs"
        self.research_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        baseline_env = _load_env_file(self.config.baseline_env_path)
        self.required_model_name = baseline_env.get("MODEL_NAME", "")
        self.required_eval_dataset_name = _infer_required_eval_dataset_name(baseline_env)

    def run(self) -> GSM8KLoopState:
        state = self._load_or_init_state()
        iteration_index = len([item for item in state.iterations if item.index >= 0])
        while self.config.max_rounds <= 0 or iteration_index < self.config.max_rounds:
            proposals = _build_dynamic_proposals(state)
            state.next_proposals = [
                {
                    "label": proposal.label,
                    "family": proposal.family,
                    "priority": proposal.priority,
                    "rationale": proposal.rationale,
                    "overrides": proposal.overrides,
                }
                for proposal in proposals[:5]
            ]
            self._write_state(state)
            self._write_director_summary(state)
            if not proposals:
                break
            proposal = proposals[0]
            if self.config.enable_research_wave:
                self._run_research_wave(state, iteration_index, proposal)
            iteration = self._run_iteration(state, iteration_index, proposal)
            state.iterations.append(iteration)
            if iteration.metric > state.best_metric:
                state.best_metric = iteration.metric
                state.best_exact_match_count = iteration.exact_match_count
                state.best_label = iteration.label
                state.best_output_dir = iteration.output_dir
                next_env = dict(state.current_env)
                next_env.update(iteration.overrides)
                state.current_env = next_env
                iteration.status = "keep"
                iteration.notes.append("improved-best")
            else:
                if iteration.status == "running":
                    iteration.status = "discard"
            state.latest_diagnosis = dict(iteration.diagnosis)
            self._write_state(state)
            self._write_director_summary(state)
            self._maybe_sync_repo(state, iteration.label)
            iteration_index += 1
        return state

    def _load_or_init_state(self) -> GSM8KLoopState:
        state_path = self.output_dir / "loop_state.json"
        if self.config.resume and state_path.exists():
            payload = _json_load(state_path)
            config_payload = payload.get("config", {}) or {}
            state = GSM8KLoopState(
                config=self.config,
                started_at=str(payload.get("started_at", utc_now())),
                best_metric=float(payload.get("best_metric", 0.0) or 0.0),
                best_exact_match_count=int(payload.get("best_exact_match_count", 0) or 0),
                best_label=str(payload.get("best_label", "baseline-env")),
                best_output_dir=str(payload.get("best_output_dir", "")),
                current_env=dict(payload.get("current_env", _load_env_file(self.config.baseline_env_path))),
                iterations=[LoopIteration(**item) for item in payload.get("iterations", [])],
                latest_diagnosis=dict(payload.get("latest_diagnosis", {}) or {}),
                next_proposals=list(payload.get("next_proposals", []) or []),
                external_history=dict(payload.get("external_history", {}) or {}),
            )
            if config_payload.get("baseline_env_path") != str(self.config.baseline_env_path):
                state.current_env = _load_env_file(self.config.baseline_env_path)
            if self.config.experiment_root is not None:
                state.external_history = _scan_experiment_root(
                    self.config.experiment_root,
                    self.config.metric_section,
                    required_model_name=self.required_model_name,
                    required_eval_dataset_name=self.required_eval_dataset_name,
                )
            return state

        base_env = _load_env_file(self.config.baseline_env_path)
        state = GSM8KLoopState(
            config=self.config,
            started_at=utc_now(),
            best_metric=0.0,
            best_exact_match_count=0,
            best_label="baseline-env",
            best_output_dir="",
            current_env=dict(base_env),
            external_history=_scan_experiment_root(
                self.config.experiment_root,
                self.config.metric_section,
                required_model_name=self.required_model_name,
                required_eval_dataset_name=self.required_eval_dataset_name,
            ),
        )
        self._write_state(state)
        baseline_iteration = self._run_iteration(
            state,
            -1,
            HypothesisProposal(
                label="baseline",
                family="baseline",
                overrides={},
                rationale="Seed baseline on the frozen 0.565-compatible configuration.",
            ),
        )
        state.iterations.append(baseline_iteration)
        state.latest_diagnosis = dict(baseline_iteration.diagnosis)
        if baseline_iteration.status != "failed":
            state.best_metric = baseline_iteration.metric
            state.best_exact_match_count = baseline_iteration.exact_match_count
            state.best_label = baseline_iteration.label
            state.best_output_dir = baseline_iteration.output_dir
            external_best = float(state.external_history.get("best_metric", 0.0) or 0.0)
            external_dir = str(state.external_history.get("best_dir", "") or "")
            if external_best > state.best_metric:
                state.best_metric = external_best
                state.best_exact_match_count = max(
                    state.best_exact_match_count,
                    int(round(external_best * 200)),
                )
                state.best_label = "external_best"
                state.best_output_dir = external_dir
            baseline_iteration.status = "keep"
            baseline_iteration.notes.append("seed-baseline")
        self._write_state(state)
        self._write_director_summary(state)
        self._maybe_sync_repo(state, "baseline")
        return state

    def _run_research_wave(self, state: GSM8KLoopState, iteration_index: int, proposal: HypothesisProposal) -> None:
        query = (
            f"{self.config.query}\n"
            f"Current best metric: {state.best_metric:.3f} ({state.best_exact_match_count}/200).\n"
            f"Next candidate proposal label: {proposal.label}.\n"
            f"Hypothesis family: {proposal.family}.\n"
            f"Rationale: {proposal.rationale}\n"
            "Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, "
            "and whether the proposal looks plausible from local experiment history."
        )
        research_output = self.research_dir / f"iter_{iteration_index:03d}_{proposal.label}"
        run_config = RunConfig(
            query=query,
            output_dir=research_output,
            search_provider="localfs",
            local_roots=self.config.local_roots,
            max_rounds=2,
            max_subquestions=4,
            max_sources_per_question=4,
        )
        AutoResearchEngine(run_config).run()

    def _run_iteration(
        self,
        state: GSM8KLoopState,
        iteration_index: int,
        proposal: HypothesisProposal,
    ) -> LoopIteration:
        label = proposal.label
        overrides = proposal.overrides
        iter_output = self.runs_dir / f"iter_{iteration_index:03d}_{label}"
        iter_output.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(state.current_env)
        env.update(overrides)
        env["OUTPUT_DIR"] = str(iter_output)

        iteration = LoopIteration(
            index=iteration_index,
            label=label,
            started_at=utc_now(),
            base_metric=state.best_metric,
            output_dir=str(iter_output),
            overrides=dict(overrides),
            family=proposal.family,
            signature=proposal.signature,
            rationale=proposal.rationale,
        )
        log_path = iter_output / "loop.log"
        with log_path.open("w", encoding="utf-8") as log_fh:
            if self.config.runner_path is not None:
                completed = subprocess.run(
                    ["bash", str(self.config.runner_path), iter_output.name],
                    cwd=str(self.config.runner_path.parent),
                    env=env,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            else:
                completed = subprocess.run(
                    ["python3", str(self.config.script_path)],
                    cwd=str(self.config.script_path.parent),
                    env=env,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        summary_path = iter_output / "run_summary.json"
        if completed.returncode == 10 and (iter_output / "early_stop.txt").exists():
            iteration.status = "early_stop"
            iteration.completed_at = utc_now()
            iteration.notes.append("runner_early_stop")
            early_stop = (iter_output / "early_stop.txt").read_text(encoding="utf-8", errors="ignore")
            iteration.notes.extend(line.strip() for line in early_stop.splitlines() if line.strip())
            return iteration
        if completed.returncode != 0 or not summary_path.exists():
            iteration.status = "failed"
            iteration.completed_at = utc_now()
            iteration.notes.append(f"returncode={completed.returncode}")
            if log_path.exists():
                log_text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
                if "outofmemoryerror" in log_text or "cuda out of memory" in log_text:
                    iteration.notes.append("resource_failure=oom")
            return iteration
        metric, count = _safe_metric(summary_path, self.config.metric_section)
        iteration.metric = metric
        iteration.exact_match_count = count
        iteration.completed_at = utc_now()
        iteration.diagnosis = _diagnose_run_summary(summary_path, self.config.metric_section)
        return iteration

    def _write_state(self, state: GSM8KLoopState) -> None:
        (self.output_dir / "loop_state.json").write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_director_summary(self, state: GSM8KLoopState) -> None:
        lines = [
            "# GSM8K Director Summary",
            "",
            f"- started_at: {state.started_at}",
            f"- best_metric: {state.best_metric:.3f}",
            f"- best_exact_match_count: {state.best_exact_match_count}",
            f"- best_label: {state.best_label}",
            f"- best_output_dir: {state.best_output_dir or '-'}",
            "",
            "## External History",
            "",
        ]
        if state.external_history:
            lines.extend(
                [
                    f"- root: {state.external_history.get('root', '-')}",
                    f"- completed_runs: {state.external_history.get('completed_runs', 0)}",
                    f"- early_stops: {state.external_history.get('early_stops', 0)}",
                    f"- best_metric: {state.external_history.get('best_metric', 0.0):.3f}",
                    f"- best_dir: {state.external_history.get('best_dir', '-') or '-'}",
                    "",
                ]
            )
        else:
            lines.extend(["- no external experiment root configured", ""])
        lines.extend([
            "## Current Diagnosis",
            "",
        ])
        diagnosis = state.latest_diagnosis or {}
        if diagnosis:
            lines.extend(
                [
                    f"- metric_section: {diagnosis.get('metric_section', '-')}",
                    f"- eval_dataset_name: {diagnosis.get('eval_dataset_name', '-') or '-'}",
                    f"- exact_match_rate: {diagnosis.get('exact_match_rate', 0.0):.3f}",
                    f"- pass1: {diagnosis.get('pass1', 0.0):.3f}",
                    f"- pass8: {diagnosis.get('pass8', 0.0):.3f}",
                    f"- selector_gap: {diagnosis.get('gap', 0.0):.3f}",
                    f"- strict_xml_rate: {diagnosis.get('strict_xml_rate', 0.0):.3f}",
                    f"- numeric_answer_rate: {diagnosis.get('numeric_answer_rate', 0.0):.3f}",
                    f"- correctness_reward_mean: {diagnosis.get('correctness_reward_mean', 0.0):.3f}",
                    f"- distance_reward_mean: {diagnosis.get('distance_reward_mean', 0.0):.3f}",
                    f"- bottom_slices: {', '.join(diagnosis.get('bottom_slices', [])) or '-'}",
                    f"- top_wrong_patterns: {', '.join(diagnosis.get('top_wrong_patterns', [])) or '-'}",
                ]
            )
        else:
            lines.append("- no diagnosis yet")
        lines.extend(["", "## Recent Iterations", ""])
        for item in state.iterations[-5:]:
            lines.append(
                f"- iter {item.index}: {item.label} [{item.family}] status={item.status} metric={item.metric:.3f} notes={'; '.join(item.notes) or '-'}"
            )
        lines.extend(["", "## Next Hypotheses", ""])
        if state.next_proposals:
            for item in state.next_proposals:
                lines.append(
                    f"- {item['label']} [{item['family']}] priority={item['priority']:.2f}: {item['rationale']}"
                )
        else:
            lines.append("- no pending hypotheses")
        (self.output_dir / "director_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _maybe_sync_repo(self, state: GSM8KLoopState, label: str) -> None:
        script = Path(self.config.sync_script)
        repo = Path(self.config.sync_repo)
        if not script.exists() or not repo.exists():
            return
        subprocess.run(
            [str(script), str(self.config.script_path.parent), str(repo), f"mar-loop-{label}"],
            check=False,
        )
