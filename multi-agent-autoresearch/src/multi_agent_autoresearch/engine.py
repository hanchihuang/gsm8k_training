from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from multi_agent_autoresearch.agents import (
    CriticAgent,
    PlannerAgent,
    ResearchAgent,
    VerifierAgent,
    WriterAgent,
    run_research_wave,
)
from multi_agent_autoresearch.models import (
    FailureEvent,
    RunArtifacts,
    RunConfig,
    WaveResult,
    utc_now,
)
from multi_agent_autoresearch.providers import build_search_provider


@dataclass(slots=True)
class FailureAttributor:
    def attribute(self, stage: str, exc: Exception) -> FailureEvent:
        message = str(exc).strip() or exc.__class__.__name__
        lowered = message.lower()
        attributed_to = stage
        if "timed out" in lowered or "http" in lowered:
            attributed_to = "research"
        elif "parse" in lowered:
            attributed_to = "writer"
        elif "empty" in lowered or "missing" in lowered:
            attributed_to = "planner"
        return FailureEvent(
            stage=stage,
            error_type=exc.__class__.__name__,
            message=message,
            attributed_to=attributed_to,
        )


class AutoResearchEngine:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.search_provider = build_search_provider(
            config.search_provider,
            roots=config.local_roots,
        )
        self.planner = PlannerAgent()
        self.researcher = ResearchAgent(self.search_provider)
        self.critic = CriticAgent()
        self.verifier = VerifierAgent()
        self.writer = WriterAgent()
        self.attributor = FailureAttributor()

    def run(self) -> RunArtifacts:
        failures: list[FailureEvent] = []
        started_at = utc_now()
        try:
            plan = self.planner.plan(self.config.query, self.config.max_subquestions)
        except Exception as exc:
            failures.append(self.attributor.attribute("planner", exc))
            raise

        waves: list[WaveResult] = []
        accumulated_evidence = []
        questions = list(plan.subquestions)
        last_critique = None

        for round_index in range(self.config.max_rounds):
            if not questions:
                break
            try:
                evidence = run_research_wave(
                    questions=questions,
                    researcher=self.researcher,
                    max_sources_per_question=self.config.max_sources_per_question,
                )
                accumulated_evidence.extend(evidence)
                critique = self.critic.review(self.config.query, accumulated_evidence, round_index)
                waves.append(
                    WaveResult(
                        round_index=round_index,
                        questions=list(questions),
                        evidence=list(evidence),
                        critique=critique,
                    )
                )
                last_critique = critique
                if critique.decision != "continue":
                    break
                questions = critique.follow_up_questions
            except Exception as exc:
                failures.append(self.attributor.attribute("research", exc))
                break

        if last_critique is None:
            last_critique = self.critic.review(self.config.query, accumulated_evidence, 99)

        try:
            claims, lessons = self.verifier.verify(self.config.query, accumulated_evidence)
            report_markdown = self.writer.write(
                query=self.config.query,
                plan=plan,
                evidence=accumulated_evidence,
                claims=claims,
                lessons=lessons,
                critique=last_critique,
            )
        except Exception as exc:
            failures.append(self.attributor.attribute("writer", exc))
            claims = []
            lessons = []
            report_markdown = f"# Research Report\n\nGeneration failed: {exc}\n"

        completed_at = utc_now()
        artifacts = RunArtifacts(
            config=self.config,
            started_at=started_at,
            completed_at=completed_at,
            plan=plan,
            waves=waves,
            claims=claims,
            lessons=lessons,
            failures=failures,
            report_markdown=report_markdown,
        )
        self._write_outputs(artifacts)
        return artifacts

    def _write_outputs(self, artifacts: RunArtifacts) -> None:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.md").write_text(artifacts.report_markdown, encoding="utf-8")
        (output_dir / "report.json").write_text(
            json.dumps(artifacts.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        trace = {
            "started_at": artifacts.started_at,
            "completed_at": artifacts.completed_at,
            "search_provider": self.search_provider.name,
            "wave_count": len(artifacts.waves),
            "claim_count": len(artifacts.claims),
            "failure_count": len(artifacts.failures),
        }
        (output_dir / "run_trace.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
