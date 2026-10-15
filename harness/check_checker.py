import argparse
import asyncio
import json
from pathlib import Path

from harness.agents import AgentRunner
from harness.budget import BudgetTracker
from harness.config import load_settings
from harness.events import EventLogger
from harness.llm import ModelClient
from harness.schemas import (
    AgentRole,
    AnalysisOutput,
    Citation,
    Finding,
    ReviewOutput,
)
from harness.tools import DocumentStore, ToolDispatcher


def load_findings(run_path: str) -> AnalysisOutput:
    path = Path(run_path)
    if path.is_dir():
        path = path / "events.jsonl"

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if (
                event["event"] == "agent_completed"
                and event["agent"] == "evidence"
            ):
                return AnalysisOutput.model_validate(
                    event["data"]["output"]
                )

    raise ValueError("No completed evidence output found in the trace")


async def main(run_path: str):
    settings = load_settings()
    store = DocumentStore("data/smoke")
    analysis = load_findings(run_path)

    # Confirm that saved citations still match the current input snapshot.
    for finding in analysis.findings:
        for citation in finding.citations:
            store.validate_citation(citation)

    injected_id = "TEST_UNSUPPORTED_AGREEMENT"
    if any(f.finding_id == injected_id for f in analysis.findings):
        raise ValueError("Injected test ID already exists")

    false_finding = Finding(
        finding_id=injected_id,
        claim="Both documents agree that the launch date is 15 November.",
        citations=[
            Citation(
                source_id="brief.txt",
                start_line=2,
                end_line=2,
                quote="The launch date is 15 November.",
            ),
            Citation(
                source_id="notes.txt",
                start_line=2,
                end_line=2,
                quote="The launch date is 22 November.",
            ),
        ],
        uncertainty="none",
    )

    # Quotes are valid, but their meaning contradicts the claim.
    for citation in false_finding.citations:
        store.validate_citation(citation)

    findings = [*analysis.findings, false_finding]

    try:
        answer = input(
            "Allow a Groq checker test on saved findings "
            "(multiple bounded requests; charges may apply)? [y/N]: "
        )
    except EOFError:
        print("No interactive input. No API request made.")
        return

    approved = answer.strip().lower() == "y"

    with EventLogger(
        "runs",
        secrets=(settings.groq_api_key.get_secret_value(),),
    ) as logger:
        logger.emit(
            "approval_decision",
            approved=approved,
            scope="checker test with a deliberately unsupported finding",
            model=settings.groq_model,
        )

        if not approved:
            print("No API request made.")
            return

        logger.emit(
            "test_fixture",
            injected_finding_id=injected_id,
            expected_verdict="reject",
            source_run=run_path,
        )

        budget = BudgetTracker(settings)
        client = ModelClient(settings, budget, logger, approved=True)
        runner = AgentRunner(client, ToolDispatcher(store))

        try:
            review = await runner.run(
                role=AgentRole.CHECKER,
                task_id="checker-smoke",
                instruction=(
                    "Review every supplied finding exactly once. "
                    "Accept a finding only if its cited sources support "
                    "its claim. A statement from one document may be "
                    "accepted as a source-specific statement even when "
                    "another document conflicts with it. Claims that "
                    "both documents agree require actual agreement. "
                    "For rejection, explain the problem and provide a "
                    "specific requested_change. Identify coverage gaps. "
                    "Do not add or alter finding IDs."
                ),
                context={
                    "goal": (
                        "Assess the documented launch dates and "
                        "first-release languages across both documents."
                    ),
                    "sources": store.manifest(),
                    "findings": [
                        finding.model_dump(mode="json")
                        for finding in findings
                    ],
                },
                output_schema=ReviewOutput,
                max_tool_rounds=4,
            )

            expected_ids = {f.finding_id for f in findings}
            actual_ids = {d.finding_id for d in review.decisions}

            if actual_ids != expected_ids:
                raise ValueError(
                    "Checker review coverage mismatch. "
                    f"Missing: {sorted(expected_ids - actual_ids)}; "
                    f"Unknown: {sorted(actual_ids - expected_ids)}"
                )

            decisions = {
                decision.finding_id: decision
                for decision in review.decisions
            }
            rejected_false_claim = (
                decisions[injected_id].verdict == "reject"
            )

            print("\nChecker output:")
            print(review.model_dump_json(indent=2))

            logger.emit(
                "checker_test_result",
                review_coverage_complete=True,
                unsupported_claim_rejected=rejected_false_claim,
            )

            if not rejected_false_claim:
                raise ValueError(
                    "Checker accepted the deliberately unsupported claim"
                )

            logger.emit(
                "run_finished",
                status="complete",
                scope="checker coverage and semantic rejection test",
            )
            print("\nChecker test passed:")
            print("- Every finding received a decision.")
            print("- The unsupported agreement claim was rejected.")
            print("- The rejection contains correction instructions.")

        except Exception as exc:
            logger.emit(
                "run_finished",
                status="failed",
                reason=type(exc).__name__,
            )
            raise

        finally:
            try:
                logger.emit(
                    "budget_snapshot",
                    snapshot=await budget.snapshot(),
                )
            finally:
                await client.close()
                print("\nTrace:", logger.run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run")
    args = parser.parse_args()
    asyncio.run(main(args.source_run))

    