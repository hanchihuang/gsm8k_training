from __future__ import annotations

import csv
import html
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol

from multi_agent_autoresearch.models import Source


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, limit: int) -> list[Source]:
        ...


@dataclass(slots=True)
class MockSearchProvider:
    name: str = "mock"

    _corpus = [
        {
            "title": "CORAL introduces durable multi-agent self-evolution",
            "url": "https://github.com/Human-Agent-Society/CORAL",
            "snippet": (
                "CORAL emphasizes isolated workspaces, persistent shared knowledge, "
                "safe evaluation, and multi-agent collaboration."
            ),
        },
        {
            "title": "autoresearch-swarm adds parallel experiment waves",
            "url": "https://github.com/rock-mind/autoresearch-swarm",
            "snippet": (
                "The swarm orchestrator assigns one agent per workspace, shares best results, "
                "stores outcomes in SQLite, and writes markdown reports."
            ),
        },
        {
            "title": "autolab focuses on judgement, steering, and multi-agent competition",
            "url": "https://github.com/dean0x/autolab",
            "snippet": (
                "autojudge replaces eyeballing with statistical verdicts, autosteer suggests "
                "what to try next, and autoevolve spreads winning ideas."
            ),
        },
        {
            "title": "Autoresearch.ai uses planner, researchers, critic, writer",
            "url": "https://github.com/manavchouhan115/Autoresearch.ai",
            "snippet": (
                "A planner decomposes a query, researcher agents gather web evidence in "
                "parallel, a critic requests follow-up questions, and a writer synthesizes."
            ),
        },
        {
            "title": "multi-autoresearch uses wave-based parallel worktrees",
            "url": "https://github.com/chrisliu298/multi-autoresearch",
            "snippet": (
                "Experiments run in isolated worktrees, winners are merged back, and multi-"
                "perspective ideation is invoked when progress stalls."
            ),
        },
        {
            "title": "failure-attribution-debugger treats root-cause identification as a metric",
            "url": "https://github.com/rambo-01/failure-attribution-debugger",
            "snippet": (
                "The system records pipeline traces and attributes which agent introduced "
                "a downstream failure, improving observability in multi-agent systems."
            ),
        },
        {
            "title": "society-autoresearch shares lessons across specialists",
            "url": "https://github.com/dimas-timmers/society-autoresearch",
            "snippet": (
                "A shared knowledge pool stores accepted and failed experiments so "
                "specialist agents compound gains instead of repeating mistakes."
            ),
        },
        {
            "title": "Most topic-research demos stop too early",
            "url": "https://example.com/research-demos-stop-early",
            "snippet": (
                "Many projects generate one report from one pass, but lack critic loops, "
                "verification, resumability, and durable lessons."
            ),
        },
    ]

    def search(self, query: str, limit: int) -> list[Source]:
        tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        ranked = []
        for item in self._corpus:
            text = f"{item['title']} {item['snippet']}".lower()
            score = sum(1 for token in tokens if token in text)
            ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [
            Source(
                title=item["title"],
                url=item["url"],
                snippet=item["snippet"],
                provider=self.name,
            )
            for score, item in ranked[:limit]
            if score > 0 or len(tokens) < 3
        ]


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[tuple[str, str]] = []
        self._capture_title = False
        self._current_href = ""
        self._current_title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        class_name = attr_map.get("class", "") or ""
        if tag == "a" and "result__a" in class_name:
            self._capture_title = True
            self._current_href = attr_map.get("href", "") or ""
            self._current_title_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_title:
            title = html.unescape("".join(self._current_title_parts)).strip()
            if title and self._current_href:
                self.results.append((title, self._current_href))
            self._capture_title = False
            self._current_href = ""
            self._current_title_parts = []

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self._current_title_parts.append(data)


@dataclass(slots=True)
class DuckDuckGoSearchProvider:
    name: str = "duckduckgo"

    def search(self, query: str, limit: int) -> list[Source]:
        encoded = urllib.parse.urlencode({"q": query})
        url = f"https://duckduckgo.com/html/?{encoded}"
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; multi-agent-autoresearch/0.1)"
            },
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8", "ignore")
        parser = _DuckDuckGoHTMLParser()
        parser.feed(body)
        sources: list[Source] = []
        for title, url in parser.results[:limit]:
            clean_url = re.sub(r"^//duckduckgo.com/l/\\?uddg=", "", url)
            clean_url = urllib.parse.unquote(clean_url)
            sources.append(
                Source(
                    title=title,
                    url=clean_url,
                    snippet=f"Search result for query: {query}",
                    provider=self.name,
                )
            )
        return sources

@dataclass(slots=True)
class LocalFileSearchProvider:
    roots: list[str]
    name: str = "localfs"
    max_file_bytes: int = 200_000
    ignored_dir_names: tuple[str, ...] = (
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
    )
    allowed_suffixes: tuple[str, ...] = (
        ".py",
        ".md",
        ".txt",
        ".tsv",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".log",
    )
    noisy_line_markers: tuple[str, ...] = (
        '"raw_completion"',
        '"question"',
        '"answer"',
        "<reasoning>",
        "<answer>",
    )
    signal_markers: tuple[str, ...] = (
        "experiment_note",
        "best_metric",
        "current_metric",
        "last_trial_metric",
        "exact_match",
        "accuracy",
        "teacher_anchor",
        "promptreplay",
        "prompt_replay",
        "masktrunc",
        "verifier",
        "rerank",
        "reward",
        "retained",
        "keep75c",
        "confirm200",
        "scout",
        "failure",
        "bottleneck",
        "metric",
    )
    prioritized_files: tuple[str, ...] = (
        "research-results.tsv",
        "autoresearch-state.json",
        "autoresearch-lessons.md",
        "run_summary.json",
    )

    def search(self, query: str, limit: int) -> list[Source]:
        tokens = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        results: list[tuple[float, Source]] = []
        for root in self.roots:
            root_path = Path(root)
            if not root_path.exists():
                continue
            for path in self._iter_files(root_path):
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if not text.strip():
                    continue
                snippet = self._structured_snippet(path, text, tokens)
                score = self._score(query, path.name, snippet)
                if score <= 0:
                    continue
                results.append(
                    (
                        score,
                        Source(
                            title=str(path.relative_to(root_path.parent)),
                            url=str(path),
                            snippet=snippet,
                            provider=self.name,
                        ),
                    )
                )
        results.sort(key=lambda item: item[0], reverse=True)
        return [source for _score, source in results[:limit]]

    def _iter_files(self, root: Path):
        if root.is_file():
            if self._allowed(root):
                yield root
            return
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in self.ignored_dir_names for part in path.parts):
                continue
            if not self._allowed(path):
                continue
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
            except OSError:
                continue
            yield path

    def _allowed(self, path: Path) -> bool:
        return path.suffix.lower() in self.allowed_suffixes

    def _structured_snippet(self, path: Path, text: str, tokens: set[str]) -> str:
        if path.name == "autoresearch-state.json":
            snippet = self._summarize_autoresearch_state(text)
            if snippet:
                return snippet
        if path.name == "research-results.tsv":
            snippet = self._summarize_research_results(text)
            if snippet:
                return snippet
        if path.name == "autoresearch-lessons.md":
            snippet = self._summarize_lessons(text)
            if snippet:
                return snippet
        return self._best_snippet(text, tokens)

    def _best_snippet(self, text: str, tokens: set[str]) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        best_line = lines[0]
        best_score = -1
        for line in lines[:600]:
            lower = line.lower()
            if any(marker in lower for marker in self.noisy_line_markers):
                continue
            score = sum(1 for token in tokens if token in lower)
            score += 2 * sum(1 for marker in self.signal_markers if marker in lower)
            if len(line) > 420:
                score -= 1
            if score > best_score:
                best_score = score
                best_line = line
        return best_line[:400]

    def _summarize_autoresearch_state(self, text: str) -> str:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return ""
        state = payload.get("state", {})
        supervisor = payload.get("supervisor", {})
        parts = [
            "autoresearch state",
            f"iteration {state.get('iteration', 'unknown')}",
            f"best_metric {state.get('best_metric', 'unknown')}",
            f"current_metric {state.get('current_metric', 'unknown')}",
            f"best_iteration {state.get('best_iteration', 'unknown')}",
            f"last_status {state.get('last_status', 'unknown')}",
            f"recommended_action {supervisor.get('recommended_action', 'unknown')}",
        ]
        reason = supervisor.get("last_reason")
        if reason:
            parts.append(f"reason {reason}")
        updated_at = payload.get("updated_at")
        if updated_at:
            parts.append(f"updated_at {updated_at}")
        return "; ".join(str(part) for part in parts if part not in {"", None})

    def _summarize_research_results(self, text: str) -> str:
        rows = [
            line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
        ]
        if not rows:
            return ""
        reader = csv.DictReader(rows, delimiter="\t")
        parsed_rows = list(reader)
        if not parsed_rows:
            return ""
        baseline = next((row for row in parsed_rows if row.get("status") == "baseline"), None)
        keep_rows = [row for row in parsed_rows if row.get("status") == "keep"]
        pivot_rows = [row for row in parsed_rows if row.get("status") == "pivot"]
        best_row = max(
            parsed_rows,
            key=lambda row: self._safe_float(row.get("metric")),
        )
        latest_row = parsed_rows[-1]
        parts = [
            "research results",
            f"baseline_metric {self._safe_float(baseline.get('metric')) if baseline else 'unknown'}",
            f"best_metric {self._safe_float(best_row.get('metric'))}",
            f"best_status {best_row.get('status', 'unknown')}",
            f"best_description {best_row.get('description', '').strip()}",
            f"keep_count {len(keep_rows)}",
        ]
        if pivot_rows:
            parts.append(f"latest_pivot {pivot_rows[-1].get('description', '').strip()}")
        if latest_row:
            parts.append(f"latest_status {latest_row.get('status', 'unknown')}")
        return "; ".join(part for part in parts if part)

    def _summarize_lessons(self, text: str) -> str:
        headings = [
            line.removeprefix("### ").strip()
            for line in text.splitlines()
            if line.startswith("### ")
        ]
        if not headings:
            return ""
        recent = headings[-3:]
        return "lessons summary; recent_lessons " + " | ".join(recent)

    def _safe_float(self, value: str | None) -> float:
        try:
            return float(value) if value not in {None, ""} else float("-inf")
        except ValueError:
            return float("-inf")

    def _score(self, query: str, title: str, snippet: str) -> float:
        q_tokens = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        text_tokens = set(re.findall(r"[a-zA-Z0-9_]+", f"{title} {snippet}".lower()))
        if not q_tokens:
            return 0.0
        overlap = len(q_tokens & text_tokens)
        boost = sum(1 for marker in self.signal_markers if marker in f"{title} {snippet}".lower())
        file_boost = 3 if any(name in title for name in self.prioritized_files) else 0
        penalty = 2 if any(marker in snippet.lower() for marker in self.noisy_line_markers) else 0
        return round((overlap + file_boost + 0.35 * boost - 0.5 * penalty) / max(len(q_tokens), 1), 3)


def build_search_provider(name: str, roots: list[str] | None = None) -> SearchProvider:
    if name == "mock":
        return MockSearchProvider()
    if name == "duckduckgo":
        return DuckDuckGoSearchProvider()
    if name == "localfs":
        if not roots:
            raise ValueError("localfs search provider requires at least one local root")
        return LocalFileSearchProvider(roots=roots)
    raise ValueError(f"Unsupported search provider: {name}")
