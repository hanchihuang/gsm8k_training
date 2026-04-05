from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class Source:
    title: str
    url: str
    snippet: str
    provider: str


@dataclass(slots=True)
class Evidence:
    query: str
    source: Source
    extracted_fact: str
    relevance: float


@dataclass(slots=True)
class Claim:
    text: str
    evidence_urls: list[str] = field(default_factory=list)
    support_score: float = 0.0
    status: str = "unverified"


@dataclass(slots=True)
class Critique:
    decision: str
    rationale: str
    follow_up_questions: list[str] = field(default_factory=list)
    missing_topics: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Lesson:
    kind: str
    text: str
    round_index: int


@dataclass(slots=True)
class FailureEvent:
    stage: str
    error_type: str
    message: str
    attributed_to: str


@dataclass(slots=True)
class ResearchPlan:
    main_question: str
    subquestions: list[str]
    hypotheses: list[str]


@dataclass(slots=True)
class WaveResult:
    round_index: int
    questions: list[str]
    evidence: list[Evidence]
    critique: Critique


@dataclass(slots=True)
class RunConfig:
    query: str
    output_dir: Path
    search_provider: str = "mock"
    local_roots: list[str] = field(default_factory=list)
    max_rounds: int = 2
    max_subquestions: int = 4
    max_sources_per_question: int = 4


@dataclass(slots=True)
class RunArtifacts:
    config: RunConfig
    started_at: str
    completed_at: str
    plan: ResearchPlan
    waves: list[WaveResult]
    claims: list[Claim]
    lessons: list[Lesson]
    failures: list[FailureEvent]
    report_markdown: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config"]["output_dir"] = str(self.config.output_dir)
        return payload
