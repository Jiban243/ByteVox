import json

from harness.events import EventLogger, show_trace
from harness.schemas import AgentRole


def test_events_are_ordered_and_secrets_are_redacted(tmp_path):
    fake_key = "test-secret-do-not-log"

    with EventLogger(tmp_path, secrets=(fake_key,)) as logger:
        logger.emit("run_started", goal="Inspect fictional documents")
        logger.emit(
            "model_error",
            agent=AgentRole.EVIDENCE,
            error=f"Example message containing {fake_key}",
            headers={"Authorization": f"Bearer {fake_key}"},
        )
        path = logger.path

    text = path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in text.splitlines()]

    assert fake_key not in text
    assert [record["sequence"] for record in records] == [1, 2]
    assert records[1]["agent"] == "evidence"
    assert records[1]["data"]["headers"]["Authorization"] == "[REDACTED]"


def test_trace_view_handles_an_incomplete_event(tmp_path, capsys):
    with EventLogger(tmp_path) as logger:
        logger.emit("run_started")
        path = logger.path

    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"unfinished":')

    show_trace(path)
    output = capsys.readouterr().out

    assert "run_started" in output
    assert "invalid or incomplete event" in output

