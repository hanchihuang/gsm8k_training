from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from multi_agent_autoresearch.models import (
    Claim,
    Critique,
    Evidence,
    Lesson,
    ResearchPlan,
)
from multi_agent_autoresearch.providers import SearchProvider


def _normalize_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())
        if token not in {"what", "which", "with", "that", "this", "from", "into", "have"}
    ]


@dataclass(slots=True)
class PlannerAgent:
    def plan(self, query: str, max_subquestions: int) -> ResearchPlan:
        seed_questions = [
            f"What sub-problems must be solved to answer: {query}?",
            f"What evidence would make the answer to '{query}' trustworthy?",
            f"What are the strongest design patterns related to: {query}?",
            f"What failure modes or blind spots appear in systems for: {query}?",
            f"What would an opinionated but practical implementation for '{query}' include?",
        ]
        hypotheses = [
            "The strongest systems combine parallelism with an explicit critic loop.",
            "Durable memory and verification matter more than fancy agent personas.",
            "Failure attribution and resumability are still underbuilt compared with report generation.",
        ]
        return ResearchPlan(
            main_question=query,
            subquestions=seed_questions[:max_subquestions],
            hypotheses=hypotheses,
        )


@dataclass(slots=True)
class ResearchAgent:
    provider: SearchProvider

    def run(self, query: str, limit: int) -> list[Evidence]:
        evidence: list[Evidence] = []
        for source in self.provider.search(query, limit=limit):
            relevance = self._score_relevance(query, source.title, source.snippet)
            extracted = self._extract_fact(source.title, source.snippet, source.provider)
            evidence.append(
                Evidence(
                    query=query,
                    source=source,
                    extracted_fact=extracted,
                    relevance=relevance,
                )
            )
        evidence.sort(key=lambda item: item.relevance, reverse=True)
        return evidence

    def _score_relevance(self, query: str, title: str, snippet: str) -> float:
        q_tokens = set(_normalize_tokens(query))
        text_tokens = set(_normalize_tokens(f"{title} {snippet}"))
        if not q_tokens:
            return 0.0
        overlap = len(q_tokens & text_tokens)
        return round(overlap / max(len(q_tokens), 1), 3)

    def _extract_fact(self, title: str, snippet: str, provider: str) -> str:
        if provider == "localfs":
            return snippet.strip()
        return f"{title}: {snippet}".strip()


@dataclass(slots=True)
class CriticAgent:
    def review(self, main_query: str, evidence: list[Evidence], round_index: int) -> Critique:
        domains = {self._domain_name(item.source.url) for item in evidence}
        repeated_tokens = Counter(
            token
            for item in evidence
            for token in _normalize_tokens(item.extracted_fact)
            if len(token) > 5
        )
        dominant_topics = [token for token, _count in repeated_tokens.most_common(6)]
        missing_topics = []
        follow_up = []
        if len(domains) < 3:
            missing_topics.append("domain-diversity")
            follow_up.append(
                f"Find more diverse sources that answer: {main_query} from different communities or toolchains."
            )
        if not any(token in dominant_topics for token in ("critic", "verify", "verification")):
            missing_topics.append("verification")
            follow_up.append(
                f"What verification and evaluation mechanisms are used in systems addressing: {main_query}?"
            )
        if not any(token in dominant_topics for token in ("memory", "knowledge", "lessons")):
            missing_topics.append("memory")
            follow_up.append(
                f"How do robust systems persist lessons, failed attempts, or shared knowledge for: {main_query}?"
            )
        decision = "continue" if follow_up else "finalize"
        rationale = (
            "The evidence is still narrow or under-verified."
            if decision == "continue"
            else "The evidence covers enough recurring patterns to synthesize."
        )
        return Critique(
            decision=decision,
            rationale=rationale,
            follow_up_questions=follow_up[:3],
            missing_topics=missing_topics,
        )

    def _domain_name(self, url: str) -> str:
        return re.sub(r"^https?://", "", url).split("/", 1)[0]


@dataclass(slots=True)
class VerifierAgent:
    def verify(self, main_query: str, evidence: list[Evidence]) -> tuple[list[Claim], list[Lesson]]:
        grouped = Counter()
        supporting_urls: dict[str, list[str]] = {}
        for item in evidence:
            for sentence in self._candidate_claims(item.extracted_fact):
                grouped[sentence] += 1
                supporting_urls.setdefault(sentence, []).append(item.source.url)
        claims: list[Claim] = []
        lessons: list[Lesson] = []
        for claim_text, count in grouped.most_common(6):
            urls = supporting_urls[claim_text]
            support_score = min(1.0, round(count / 3.0, 2))
            status = "supported" if support_score >= 0.66 else "weak"
            claims.append(
                Claim(
                    text=claim_text,
                    evidence_urls=urls[:4],
                    support_score=support_score,
                    status=status,
                )
            )
            if status == "supported":
                lessons.append(
                    Lesson(
                        kind="accepted-pattern",
                        text=claim_text,
                        round_index=0,
                    )
                )
        if not claims:
            claims.append(
                Claim(
                    text=f"No stable claims could be extracted for: {main_query}",
                    support_score=0.0,
                    status="weak",
                )
            )
        return claims, lessons

    def _candidate_claims(self, text: str) -> list[str]:
        clauses = re.split(r"(?:\n+|[.!?]\s+)", text)
        return [clause.strip() for clause in clauses if len(clause.strip().split()) >= 6]


@dataclass(slots=True)
class WriterAgent:
    def write(
        self,
        query: str,
        plan: ResearchPlan,
        evidence: list[Evidence],
        claims: list[Claim],
        lessons: list[Lesson],
        critique: Critique,
    ) -> str:
        top_sources = []
        seen = set()
        for item in evidence:
            if item.source.url in seen:
                continue
            seen.add(item.source.url)
            top_sources.append(f"- [{item.source.title}]({item.source.url})")
            if len(top_sources) == 8:
                break

        claim_lines = []
        for claim in claims:
            urls = ", ".join(claim.evidence_urls[:3]) or "n/a"
            claim_lines.append(
                f"- {claim.text}  \n  status: `{claim.status}` | support: `{claim.support_score}` | evidence: {urls}"
            )

        lesson_lines = [f"- {lesson.text}" for lesson in lessons[:8]]
        subquestion_lines = [f"- {question}" for question in plan.subquestions]

        return "\n".join(
            [
                f"# Research Report",
                "",
                f"## Query",
                "",
                query,
                "",
                "## Plan",
                "",
                *subquestion_lines,
                "",
                "## Verified Claims",
                "",
                *claim_lines,
                "",
                "## Critic Verdict",
                "",
                f"- decision: `{critique.decision}`",
                f"- rationale: {critique.rationale}",
                *(f"- follow-up: {item}" for item in critique.follow_up_questions),
                "",
                "## Lessons",
                "",
                *lesson_lines,
                "",
                "## Sources",
                "",
                *top_sources,
            ]
        )


def run_research_wave(
    questions: list[str],
    researcher: ResearchAgent,
    max_sources_per_question: int,
) -> list[Evidence]:
    with ThreadPoolExecutor(max_workers=max(1, len(questions))) as executor:
        futures = [
            executor.submit(researcher.run, question, max_sources_per_question)
            for question in questions
        ]
        evidence: list[Evidence] = []
        for future in futures:
            evidence.extend(future.result())
    evidence.sort(key=lambda item: item.relevance, reverse=True)
    return evidence
