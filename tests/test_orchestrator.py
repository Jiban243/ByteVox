import json

import pytest

from harness.budget import BudgetTracker
from harness.config import Settings
from harness.events import EventLogger
from harness.orchestrator import Orchestrator
from harness.schemas import (
    AgentRole,
    AnalysisOutput,
    Assignment,
    Citation,
    FinalSelection,
    Finding,
    Plan,
    ReviewDecision,
    ReviewOutput,
)
from harness.tools import DocumentStore


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    ["revision_success", "revision_limit", "specialist_failure"],
)
async def test_orchestration_controls(tmp_path, scenario):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "brief.txt").write_text(
        "The launch date is 10 December.\n",
        encoding="utf-8",
    )

    settings = Settings(
        groq_api_key="test-only",
        groq_model="test-only",
    )
    store = DocumentStore(input_dir)
    budget = BudgetTracker(settings)

    calls = []
    review_count = 0

    with EventLogger(tmp_path / "runs") as logger:
        orchestrator = Orchestrator(
            settings=settings,
            store=store,
            logger=logger,
            client=None,
            budget=budget,
        )

        # Scripted agent outputs exercise the real orchestration code.
        # This does not test model quality or make network requests.
        async def fake_invoke(
            role,
            task_id,
            instruction,
            context,
            schema,
            *,
            tool_rounds=4,
        ):
            nonlocal review_count
            calls.append((role, task_id))

            if schema is Plan:
                return Plan(
                    goal=context["goal"],
                    assignments=[
                        Assignment(
                            task_id="extract",
                            role=AgentRole.EVIDENCE,
                            objective="Extract the launch date.",
                            source_ids=["brief.txt"],
                            acceptance_criteria=["Cite the source."],
                        ),
                        Assignment(
                            task_id="compare",
                            role=AgentRole.CONSISTENCY,
                            objective="Check source consistency.",
                            source_ids=["brief.txt"],
                            acceptance_criteria=["Use source evidence."],
                        ),
                    ],
                )

            if schema is AnalysisOutput:
                if (
                    scenario == "specialist_failure"
                    and role == AgentRole.EVIDENCE
                ):
                    raise TimeoutError("Injected specialist timeout")

                return AnalysisOutput(
                    findings=[
                        Finding(
                            finding_id="local-F1",
                            claim="The launch date is 10 December.",
                            citations=[
                                Citation(
                                    source_id="brief.txt",
                                    start_line=1,
                                    end_line=1,
                                    quote="The launch date is 10 December.",
                                )
                            ],
                            uncertainty="",
                        )
                    ],
                    unresolved_questions=[],
                )

            if schema is ReviewOutput:
                review_count += 1
                decisions = []

                for finding in context["findings"]:
                    is_evidence = finding["finding_id"].startswith(
                        "evidence:"
                    )

                    reject = is_evidence and (
                        scenario == "revision_limit"
                        or (
                            scenario == "revision_success"
                            and review_count == 1
                        )
                    )

                    decisions.append(ReviewDecision(
                        finding_id=finding["finding_id"],
                        verdict="reject" if reject else "accept",
                        reason=(
                            "Scripted rejection to exercise revision."
                            if reject
                            else "Scripted acceptance."
                        ),
                        requested_change=(
                            "Recheck and resubmit the supported finding."
                            if reject
                            else None
                        ),
                    ))

                return ReviewOutput(
                    decisions=decisions,
                    coverage_gaps=[],
                )

            if schema is FinalSelection:
                # Deliberately omit all accepted IDs.
                # The harness must restore them deterministically.
                return FinalSelection(ordered_finding_ids=[])

            raise AssertionError(f"Unexpected schema: {schema}")

        orchestrator.invoke = fake_invoke

        result = await orchestrator.execute("Report the launch date.")

        events = [
            json.loads(line)
            for line in logger.path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]

        accepted_ids = {
            finding["finding_id"]
            for finding in result["accepted_findings"]
        }
        review_calls = [
            task_id
            for role, task_id in calls
            if role == AgentRole.CHECKER
        ]

        # Every scenario reaches assembly with at least one accepted finding.
        assert accepted_ids
        assert any(
            event["event"] == "assembly_omissions_restored"
            for event in events
        )

        # The persisted state and report must match the returned result.
        saved = json.loads(
            (logger.run_dir / "state.json").read_text(
                encoding="utf-8"
            )
        )
        assert saved["status"] == result["status"]
        assert saved["accepted_findings"] == result["accepted_findings"]

        report = (logger.run_dir / "report.md").read_text(
            encoding="utf-8"
        )
        assert f"**{result['status'].upper()}**" in report

        if scenario == "revision_success":
            assert result["status"] == "complete"
            assert review_count == 2
            assert "evidence:v1:F1" in accepted_ids
            assert "evidence:v0:F1" not in accepted_ids
            assert "consistency:v0:F1" in accepted_ids
            assert not result["unverified_findings"]

            # Only the rejected specialist should be revised.
            assert (AgentRole.EVIDENCE, "extract:v1") in calls
            assert (AgentRole.CONSISTENCY, "compare:v1") not in calls

        elif scenario == "revision_limit":
            assert result["status"] == "partial"
            assert review_count == 2
            assert review_calls == ["review:v0", "review:v1"]
            assert not any(":v2" in task_id for _, task_id in calls)

            assert accepted_ids == {"consistency:v0:F1"}
            assert {
                finding["finding_id"]
                for finding in result["unverified_findings"]
            } == {"evidence:v1:F1"}

            assert any(
                "revision limit" in limitation
                for limitation in result["limitations"]
            )

        else:
            assert result["status"] == "partial"
            assert review_count == 1
            assert accepted_ids == {"consistency:v0:F1"}
            assert result["tasks"]["evidence"]["status"] == "failed"
            assert any(
                "evidence initial analysis failed" in limitation
                for limitation in result["limitations"]
            )

