import argparse
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.schemas import AgentRole


class EventLogger:
    """Write one JSON record per event into a unique run directory."""

    def __init__(
        self,
        output_root: str | Path = "runs",
        secrets: tuple[str, ...] = (),
    ):
        self.run_id = uuid.uuid4().hex
        self.run_dir = Path(output_root).resolve() / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.path = self.run_dir / "events.jsonl"
        self._handle = self.path.open("x", encoding="utf-8")
        self._started = time.monotonic()
        self._sequence = 0
        self._lock = threading.Lock()

        # Longest first avoids partially masking overlapping secrets.
        self._secrets = sorted(
            {secret for secret in secrets if secret},
            key=len,
            reverse=True,
        )

    def _redact_text(self, value: str) -> str:
        for secret in self._secrets:
            value = value.replace(secret, "[REDACTED]")
        return value

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact_text(value)

        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                key = str(key)
                normalized = key.casefold()
                sensitive = (
                    "api_key" in normalized
                    or "authorization" in normalized
                    or "password" in normalized
                    or normalized in {"secret", "access_token", "refresh_token"}
                )

                safe_key = self._redact_text(key)
                result[safe_key] = (
                    "[REDACTED]" if sensitive else self._sanitize(item)
                )
            return result

        if isinstance(value, (list, tuple)):
            return [self._sanitize(item) for item in value]

        if value is None or isinstance(value, (bool, int, float)):
            return value

        # Avoid implicitly stringifying clients or configuration objects.
        raise TypeError(
            f"Unsupported event value type: {type(value).__name__}"
        )

    def emit(
        self,
        event: str,
        *,
        agent: AgentRole | None = None,
        task_id: str | None = None,
        **data: Any,
    ) -> dict:
        with self._lock:
            if self._handle.closed:
                raise RuntimeError("Event logger is closed")

            record = self._sanitize({
                "sequence": self._sequence + 1,
                "run_id": self.run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(
                    time.monotonic() - self._started, 6
                ),
                "event": event,
                "agent": agent.value if agent is not None else None,
                "task_id": task_id,
                "data": data,
            })

            encoded = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
            )
            self._handle.write(encoded + "\n")
            self._handle.flush()
            self._sequence += 1
            return record

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def show_trace(path: str | Path, full: bool = False) -> None:
    trace_path = Path(path)
    if trace_path.is_dir():
        trace_path = trace_path / "events.jsonl"

    with trace_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Line {line_number}: invalid or incomplete event")
                continue

            # JSON escaping keeps untrusted terminal control characters
            # from being printed as active escape sequences.
            agent = json.dumps(record.get("agent"), ensure_ascii=True)
            event = json.dumps(record.get("event"), ensure_ascii=True)

            print(
                f"{record['sequence']:04d} "
                f"+{record['elapsed_seconds']:.3f}s "
                f"{agent} {event}"
            )

            if full:
                print(json.dumps(
                    record["data"],
                    indent=2,
                    ensure_ascii=True,
                ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a harness trace")
    parser.add_argument("path", help="Run directory or events.jsonl path")
    parser.add_argument("--full", action="store_true")
    arguments = parser.parse_args()

    try:
        show_trace(arguments.path, full=arguments.full)
    except OSError as exc:
        parser.exit(
            status=1,
            message=f"Cannot read trace: {exc}\n",
        )



