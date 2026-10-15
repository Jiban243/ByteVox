import asyncio

from harness.agents import AgentRunner
from harness.budget import BudgetTracker
from harness.config import load_settings
from harness.events import EventLogger
from harness.llm import ModelClient
from harness.schemas import AgentRole, AnalysisOutput
from harness.tools import DocumentStore, ToolDispatcher, ToolError


async def main():
    settings = load_settings()
    store = DocumentStore("data/smoke")

    try:
        answer = input(
            "Allow one evidence-agent test using Groq "
            "(multiple bounded requests; charges may apply)? [y/N]: "
        )
    except EOFError:
        print("No interactive input available. No API request made.")
        return

    approved = answer.strip().lower() == "y"

    with EventLogger(
        "runs",
        secrets=(settings.groq_api_key.get_secret_value(),),
    ) as logger:
        logger.emit(
            "approval_decision",
            approved=approved,
            scope="one evidence specialist test",
            model=settings.groq_model,
        )

        if not approved:
            print("No API request made.")
            return

        budget = BudgetTracker(settings)
        client = ModelClient(
            settings,
            budget,
            logger,
            approved=True,
        )
        runner = AgentRunner(client, ToolDispatcher(store))

        try:
            output = await runner.run(
                role=AgentRole.EVIDENCE,
                task_id="evidence-smoke",
                instruction=(
                    "Extract explicit launch-date and supported-language "
                    "statements from BOTH supplied documents. "
                    "The context already contains the complete source "
                    "manifest, so do not list documents again. "
                    "Both documents are short: use read_document directly "
                    "to read each document from line 1 through its listed "
                    "line_count. Search is unnecessary for this task. "
                    "Read both documents before finishing evidence "
                    "collection. You may request both reads in one "
                    "response or read them sequentially. "
                    "Keep conflicting statements separate and identify "
                    "which source states each. Use distinct finding IDs. "
                    "Cite exact source lines for every finding. "
                    "If a document cannot be read, explicitly identify "
                    "the missing evidence in unresolved_questions. "
                    "Return an AnalysisOutput."
                ),
                context={
                    "goal": (
                        "Identify the documented launch dates and "
                        "first-release languages across both documents."
                    ),
                    "sources": store.manifest(),
                },
                output_schema=AnalysisOutput,
                max_tool_rounds=4,
            )

            print("\nAgent output:")
            print(output.model_dump_json(indent=2))

            valid_ids = set()
            invalid_findings = []

            for finding in output.findings:
                errors = []

                for citation in finding.citations:
                    try:
                        store.validate_citation(citation)
                    except ToolError as exc:
                        errors.append(str(exc))

                if errors:
                    invalid_findings.append({
                        "finding_id": finding.finding_id,
                        "errors": errors,
                    })
                else:
                    valid_ids.add(finding.finding_id)

            logger.emit(
                "citation_validation",
                agent=AgentRole.EVIDENCE,
                task_id="evidence-smoke",
                valid_finding_ids=sorted(valid_ids),
                invalid_findings=invalid_findings,
            )

            # Fixture-specific expected evidence.
            # The agent is not given these expected answers.
            expected = {
                ("brief.txt", 2),
                ("brief.txt", 3),
                ("notes.txt", 2),
                ("notes.txt", 3),
            }

            covered = {
                (citation.source_id, line_number)
                for finding in output.findings
                if finding.finding_id in valid_ids
                for citation in finding.citations
                for line_number in range(
                    citation.start_line,
                    citation.end_line + 1,
                )
            }

            missing = sorted(expected - covered)

            logger.emit(
                "fixture_coverage",
                expected=[
                    {"source_id": source_id, "line": line}
                    for source_id, line in sorted(expected)
                ],
                missing=[
                    {"source_id": source_id, "line": line}
                    for source_id, line in missing
                ],
            )

            print(f"\nFindings with valid citations: {len(valid_ids)}")
            print(
                "Findings with invalid citations: "
                f"{len(invalid_findings)}"
            )

            if invalid_findings:
                logger.emit(
                    "run_finished",
                    status="failed",
                    reason="Exact citation validation failed",
                )
                print("\nSmoke test FAILED: invalid citations.")

                for finding in invalid_findings:
                    print(
                        f"- {finding['finding_id']}: "
                        + "; ".join(finding["errors"])
                    )

                if missing:
                    print("Missing fixture evidence:", missing)

                return

            if missing:
                logger.emit(
                    "run_finished",
                    status="partial",
                    reason="Required fixture evidence is missing",
                )
                print("\nSmoke test INCOMPLETE: missing evidence.")

                for source_id, line in missing:
                    print(f"- {source_id}, line {line}")

                return

            logger.emit(
                "run_finished",
                status="complete",
                scope=(
                    "Specialist citation and fixture-evidence coverage "
                    "check; semantic review not performed"
                ),
            )

            print("\nCitation and fixture-evidence coverage checks passed.")
            print(
                "Claim meaning and semantic completeness still require "
                "checker review."
            )

        except Exception as exc:
            logger.emit(
                "check_failed",
                error_type=type(exc).__name__,
            )
            logger.emit(
                "run_finished",
                status="failed",
                reason=f"Check raised {type(exc).__name__}",
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
    asyncio.run(main())

