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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config"]["output_dir"] = str(self.config.output_dir)
        payload["config"]["baseline_env_path"] = str(self.config.baseline_env_path)
        payload["config"]["script_path"] = str(self.config.script_path)
        return payload


def _proposal_ladder() -> list[tuple[str, dict[str, str]]]:
    return [
        ("numc12_top_p095", {"EVAL_NUM_CANDIDATES": "12", "EVAL_RERANK_TOP_P": "0.95"}),
        ("numc12_temp07_top_p095", {"EVAL_NUM_CANDIDATES": "12", "EVAL_RERANK_TEMPERATURE": "0.7", "EVAL_RERANK_TOP_P": "0.95"}),
        ("numc12_verifier025", {"EVAL_NUM_CANDIDATES": "12", "VERIFIER_SCORE_WEIGHT": "0.25"}),
        ("numc12_top_p095_ver025", {"EVAL_NUM_CANDIDATES": "12", "EVAL_RERANK_TOP_P": "0.95", "VERIFIER_SCORE_WEIGHT": "0.25"}),
        ("numc12_temp07_ver025", {"EVAL_NUM_CANDIDATES": "12", "EVAL_RERANK_TEMPERATURE": "0.7", "VERIFIER_SCORE_WEIGHT": "0.25"}),
        ("numc12_temp07_top_p095_ver025", {"EVAL_NUM_CANDIDATES": "12", "EVAL_RERANK_TEMPERATURE": "0.7", "EVAL_RERANK_TOP_P": "0.95", "VERIFIER_SCORE_WEIGHT": "0.25"}),
        ("numc12_verifier03", {"EVAL_NUM_CANDIDATES": "12", "VERIFIER_SCORE_WEIGHT": "0.3"}),
        ("numc12_top_p095_ver03", {"EVAL_NUM_CANDIDATES": "12", "EVAL_RERANK_TOP_P": "0.95", "VERIFIER_SCORE_WEIGHT": "0.3"}),
    ]


class GSM8KLoopRunner:
    def __init__(self, config: GSM8KLoopConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.research_dir = self.output_dir / "research"
        self.runs_dir = self.output_dir / "runs"
        self.research_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> GSM8KLoopState:
        base_env = _load_env_file(self.config.baseline_env_path)
        state = GSM8KLoopState(
            config=self.config,
            started_at=utc_now(),
            best_metric=0.0,
            best_exact_match_count=0,
            best_label="baseline-env",
            best_output_dir="",
            current_env=dict(base_env),
        )
        self._write_state(state)

        baseline_iteration = self._run_iteration(
            state,
            -1,
            "baseline",
            {},
        )
        state.iterations.append(baseline_iteration)
        if baseline_iteration.status != "failed":
            state.best_metric = baseline_iteration.metric
            state.best_exact_match_count = baseline_iteration.exact_match_count
            state.best_label = baseline_iteration.label
            state.best_output_dir = baseline_iteration.output_dir
            baseline_iteration.status = "keep"
            baseline_iteration.notes.append("seed-baseline")
        self._write_state(state)
        self._maybe_sync_repo(state, "baseline")

        proposals = _proposal_ladder()
        iteration_index = 0
        while self.config.max_rounds <= 0 or iteration_index < self.config.max_rounds:
            proposal_label, overrides = proposals[iteration_index % len(proposals)]
            self._run_research_wave(state, iteration_index, proposal_label)
            iteration = self._run_iteration(state, iteration_index, proposal_label, overrides)
            state.iterations.append(iteration)
            if iteration.metric > state.best_metric:
                state.best_metric = iteration.metric
                state.best_exact_match_count = iteration.exact_match_count
                state.best_label = iteration.label
                state.best_output_dir = iteration.output_dir
                next_env = dict(state.current_env)
                next_env.update(overrides)
                state.current_env = next_env
                iteration.status = "keep"
                iteration.notes.append("improved-best")
            else:
                iteration.status = "discard"
            self._write_state(state)
            self._maybe_sync_repo(state, iteration.label)
            iteration_index += 1
        return state

    def _run_research_wave(self, state: GSM8KLoopState, iteration_index: int, proposal_label: str) -> None:
        query = (
            f"{self.config.query}\n"
            f"Current best metric: {state.best_metric:.3f} ({state.best_exact_match_count}/200).\n"
            f"Next candidate proposal label: {proposal_label}.\n"
            "Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, "
            "and whether the proposal looks plausible from local experiment history."
        )
        research_output = self.research_dir / f"iter_{iteration_index:03d}_{proposal_label}"
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
        label: str,
        overrides: dict[str, str],
    ) -> LoopIteration:
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
        )
        log_path = iter_output / "loop.log"
        with log_path.open("w", encoding="utf-8") as log_fh:
            completed = subprocess.run(
                ["python3", str(self.config.script_path)],
                cwd=str(self.config.script_path.parent),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                check=False,
            )
        summary_path = iter_output / "run_summary.json"
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
        return iteration

    def _write_state(self, state: GSM8KLoopState) -> None:
        (self.output_dir / "loop_state.json").write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _maybe_sync_repo(self, state: GSM8KLoopState, label: str) -> None:
        script = Path(self.config.sync_script)
        repo = Path(self.config.sync_repo)
        if not script.exists() or not repo.exists():
            return
        subprocess.run(
            [str(script), str(self.config.script_path.parent), str(repo), f"mar-loop-{label}"],
            check=False,
        )
