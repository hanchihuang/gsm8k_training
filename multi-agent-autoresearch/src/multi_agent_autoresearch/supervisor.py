from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from multi_agent_autoresearch.gsm8k_loop import GSM8KLoopConfig, GSM8KLoopRunner, GSM8KLoopState
from multi_agent_autoresearch.models import utc_now


def _pid_is_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    proc_status = Path(f"/proc/{pid}/status")
    if proc_status.exists():
        try:
            for line in proc_status.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("State:"):
                    # Treat zombies as completed so the supervisor can hand off.
                    return "\tZ" not in line and "(zombie)" not in line.lower()
        except OSError:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass(slots=True)
class GSM8KSupervisorConfig:
    wait_pid: int | None
    loop_config: GSM8KLoopConfig
    poll_seconds: float = 5.0


@dataclass(slots=True)
class GSM8KSupervisorState:
    started_at: str
    completed_at: str | None
    wait_pid: int | None
    launch_started_at: str | None
    loop_state: GSM8KLoopState | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.loop_state is not None:
            payload["loop_state"] = self.loop_state.to_dict()
        return payload


class GSM8KSupervisorRunner:
    def __init__(self, config: GSM8KSupervisorConfig) -> None:
        self.config = config
        self.output_dir = config.loop_config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / "supervisor_state.json"

    def run(self) -> GSM8KSupervisorState:
        state = GSM8KSupervisorState(
            started_at=utc_now(),
            completed_at=None,
            wait_pid=self.config.wait_pid,
            launch_started_at=None,
            loop_state=None,
        )
        self._write_state(state)

        while _pid_is_running(self.config.wait_pid):
            time.sleep(max(0.1, self.config.poll_seconds))

        state.launch_started_at = utc_now()
        self._write_state(state)

        loop_state = GSM8KLoopRunner(self.config.loop_config).run()
        state.loop_state = loop_state
        state.completed_at = utc_now()
        self._write_state(state)
        return state

    def _write_state(self, state: GSM8KSupervisorState) -> None:
        self.state_path.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
