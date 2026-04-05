from __future__ import annotations

import argparse
import json
from pathlib import Path

from multi_agent_autoresearch.engine import AutoResearchEngine
from multi_agent_autoresearch.gsm8k_loop import GSM8KLoopConfig, GSM8KLoopRunner
from multi_agent_autoresearch.models import RunConfig


LANDSCAPE_SNAPSHOT = [
    "Human-Agent-Society/CORAL",
    "rock-mind/autoresearch-swarm",
    "dean0x/autolab",
    "deva-harsha-v/AutoResearch-MultiAgent",
    "PavanKAgnihotri/AutoResearchLab_MultiAgentAI",
    "harishchaurasia/multi-agent-autoresearch",
    "hanchihuang/multi-agent-autoresearch",
    "FraidoonOmarzai/AutoResearcher",
    "devadharshan11-design/AutoResearcher",
    "AtlasMindAI/AutoLab",
    "chrisliu298/multi-autoresearch",
    "Tanmay1112004/AutoResearch-AI---Multi-Agent-Autonomous-Research-System",
    "christinetyip/autoresearch-at-home-reports",
    "AyushKumar-Singh/AutoResearch-AI-Multi-Agent-LLM-Research-Automation-Platform",
    "wildhash/autoresearch-lab",
    "djk2017-Rocky/RalphHub",
    "rayklanderman/CapstoneProject-Autoresearcher",
    "manavchouhan115/Autoresearch.ai",
    "dimas-timmers/society-autoresearch",
    "zhongjiaqi2002/AutoResearch-Agent",
    "Omkar0612/AutoResearchBot",
    "vikashmehta292511/autoresearch-lab",
    "AmanChourasia7/autoresearch-lab",
    "rambo-01/failure-attribution-debugger",
    "Vikaash-dev/Autoresearch-v2",
    "zabarich/social-sim-study",
    "keonhee3337-art/sme-diagnostic-ai",
    "Techknowmadlabs/cortex-research-suite",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mar", description="Multi-agent autoresearch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the research engine")
    run_parser.add_argument("--query", required=True, help="Top-level research question")
    run_parser.add_argument("--output-dir", required=True, type=Path, help="Artifact directory")
    run_parser.add_argument(
        "--search-provider",
        default="mock",
        choices=["mock", "duckduckgo", "localfs"],
        help="Evidence retrieval backend",
    )
    run_parser.add_argument(
        "--local-root",
        action="append",
        default=[],
        help="Local file or directory roots for the localfs provider; repeatable",
    )
    run_parser.add_argument("--max-rounds", default=2, type=int)
    run_parser.add_argument("--max-subquestions", default=4, type=int)
    run_parser.add_argument("--max-sources-per-question", default=4, type=int)
    run_parser.add_argument(
        "--offline",
        action="store_true",
        help="Shortcut for --search-provider mock",
    )

    landscape_parser = subparsers.add_parser(
        "landscape", help="Write the repo landscape snapshot to Markdown"
    )
    landscape_parser.add_argument("--output", required=True, type=Path)

    loop_parser = subparsers.add_parser(
        "gsm8k-loop",
        help="Run a perpetual local GSM8K experiment loop around a frozen baseline",
    )
    loop_parser.add_argument("--output-dir", required=True, type=Path)
    loop_parser.add_argument("--baseline-env", required=True, type=Path)
    loop_parser.add_argument("--script-path", required=True, type=Path)
    loop_parser.add_argument(
        "--local-root",
        action="append",
        default=[],
        help="Local roots for the research wave; repeatable",
    )
    loop_parser.add_argument(
        "--query",
        default="How should the GSM8K confirm200 line improve beyond the current retained baseline?",
    )
    loop_parser.add_argument(
        "--metric-section",
        default="eval_before",
        choices=["eval_before", "eval_after", "eval_warmup"],
    )
    loop_parser.add_argument(
        "--iterations",
        default=0,
        type=int,
        help="0 means keep iterating until externally stopped",
    )
    loop_parser.add_argument(
        "--sync-script",
        default="/home/user/图片/gsm8k_improved/sync_experiment_to_git_repo.sh",
    )
    loop_parser.add_argument(
        "--sync-repo",
        default="/home/user/图片/gsm8k_training_repo",
    )

    return parser


def write_landscape(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# multi-agent autoresearch landscape",
        "",
        "This file captures the exact repositories returned by GitHub search for `multi-agent autoresearch` on 2026-04-04.",
        "",
        "## Distilled conclusions",
        "",
        "- Keep: planner/researcher/critic loops, parallel waves, durable lessons, verification, and failure attribution.",
        "- Avoid: one-shot report generators that skip memory, critique, and explicit acceptance criteria.",
        "- Build local-first artifacts so the system is inspectable even without a hosted dashboard.",
        "",
        "## Exact repository set",
        "",
    ]
    lines.extend(
        f"{index}. https://github.com/{repo}"
        for index, repo in enumerate(LANDSCAPE_SNAPSHOT, start=1)
    )
    lines.extend(
        [
            "",
            "## What informed this repository",
            "",
            "- `CORAL`: orchestration, durability, shared knowledge",
            "- `autoresearch-swarm`: parallel waves and shared results",
            "- `autolab`: judgement, steering, multi-agent competition",
            "- `multi-autoresearch`: worktree-style wave orchestration",
            "- `Autoresearch.ai`: planner -> researchers -> critic -> writer graph",
            "- `failure-attribution-debugger`: explicit blame assignment for pipeline failures",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "landscape":
        write_landscape(args.output)
        print(f"Wrote {args.output}")
        return

    if args.command == "gsm8k-loop":
        config = GSM8KLoopConfig(
            query=args.query,
            output_dir=args.output_dir,
            baseline_env_path=args.baseline_env,
            script_path=args.script_path,
            local_roots=list(args.local_root),
            metric_section=args.metric_section,
            max_rounds=args.iterations,
            sync_script=args.sync_script,
            sync_repo=args.sync_repo,
        )
        state = GSM8KLoopRunner(config).run()
        summary = {
            "output_dir": str(config.output_dir),
            "best_metric": state.best_metric,
            "best_exact_match_count": state.best_exact_match_count,
            "best_label": state.best_label,
            "iterations": len(state.iterations),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    search_provider = "mock" if args.offline else args.search_provider
    config = RunConfig(
        query=args.query,
        output_dir=args.output_dir,
        search_provider=search_provider,
        local_roots=list(args.local_root),
        max_rounds=args.max_rounds,
        max_subquestions=args.max_subquestions,
        max_sources_per_question=args.max_sources_per_question,
    )
    artifacts = AutoResearchEngine(config).run()
    summary = {
        "output_dir": str(config.output_dir),
        "waves": len(artifacts.waves),
        "claims": len(artifacts.claims),
        "failures": len(artifacts.failures),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
