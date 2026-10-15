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
    FinalSelection,
    Plan,
    ReviewOutput,
)
from harness.tools import DocumentStore, ToolDispatcher


class Orchestrator:
    def __init__(self, settings, store, logger, client, budget):
        self.settings = settings
        self.store = store
        self.logger = logger
        self.client = client
        self.budget = budget
        self.runner = AgentRunner(client, ToolDispatcher(store))

        # This is the authoritative shared state for this run.
        self.state = {
            "run_id": logger.run_id,
            "goal": "",
            "status": "running",
            "plan": None,
            "tasks": {},
            "reviews": [],
            "accepted_findings": [],
            "unverified_findings": [],
            "limitations": [],
        }

    def save_state(self):
        path = self.logger.run_dir / "state.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    async def invoke(
        self,
        role,
        task_id,
        instruction,
        context,
        schema,
        *,
        tool_rounds=4,
    ):
        remaining = await self.budget.remaining_seconds(role)
        if remaining <= 0:
            raise TimeoutError(f"{role.value} has no time remaining")

        return await asyncio.wait_for(
            self.runner.run(
                role=role,
                task_id=task_id,
                instruction=instruction,
                context=context,
                output_schema=schema,
                max_tool_rounds=tool_rounds,
            ),
            timeout=remaining,
        )

    async def analyze(self, assignment, revision=0, feedback=None):
        role = assignment.role
        key = role.value
        previous = self.state["tasks"].get(key, {}).get("output")

        self.state["tasks"][key] = {
            "task_id": assignment.task_id,
            "revision": revision,
            "status": "running",
            "output": previous,
            "error": None,
        }
        self.save_state()

        source_ids = set(assignment.source_ids)
        manifest = [
            item for item in self.store.manifest()
            if item["source_id"] in source_ids
        ]

        try:
            output = await self.invoke(
                role,
                f"{assignment.task_id}:v{revision}",
                (
                    assignment.objective
                    + "\nMeet these acceptance criteria:\n"
                    + "\n".join(assignment.acceptance_criteria)
                    + "\nReturn supported findings with exact citations. "
                    "Report missing evidence in unresolved_questions. "
                    "When revising, return the full corrected output, "
                    "including findings that remain valid."
                ),
                {
                    "goal": self.state["goal"],
                    "objective": assignment.objective,
                    "acceptance_criteria": assignment.acceptance_criteria,
                    "sources": manifest,
                    "previous_output": previous,
                    "review_feedback": feedback,
                },
                AnalysisOutput,
            )

            # Namespace IDs so concurrent agents cannot collide.
            for number, finding in enumerate(output.findings, start=1):
                finding.finding_id = f"{key}:v{revision}:F{number}"
                for citation in finding.citations:
                    if citation.source_id not in source_ids:
                        raise ValueError(
                            "Finding cites a source outside its assignment"
                        )
                    self.store.validate_citation(citation)

            self.state["tasks"][key].update({
                "status": "completed",
                "output": output.model_dump(mode="json"),
                "error": None,
            })

        except Exception as exc:
            # Preserve any earlier output, but do not claim revision success.
            self.state["tasks"][key].update({
                "status": "failed",
                "error": type(exc).__name__,
            })
            self.logger.emit(
                "specialist_failed",
                agent=role,
                task_id=assignment.task_id,
                revision=revision,
                error_type=type(exc).__name__,
            )

        self.save_state()

    def findings(self):
        return [
            finding
            for task in self.state["tasks"].values()
            if task["output"] is not None
            for finding in task["output"]["findings"]
        ]

    async def review(self, findings, revision):
        result = await self.invoke(
            AgentRole.CHECKER,
            f"review:v{revision}",
            (
                "Review every supplied finding exactly once. Verify "
                "semantic support using source reads. Accept only "
                "supported, relevant claims. Reject unsupported claims "
                "with specific correction instructions. Assess coverage "
                "across ALL findings collectively. "
                "An explicit comparison of conflicting source statements "
                "counts as explaining a disagreement. Do not demand a "
                "separate prose synthesis when the findings already make "
                "the comparison clear. Do not require an explanation of "
                "why sources differ unless the user explicitly asks for "
                "causes; never require speculation unsupported by sources. "
                "Report a coverage gap only when you can name a specific "
                "unanswered part of the user's goal. Do not report vague "
                "formatting or stylistic preferences as missing coverage. "
                "Preserve finding IDs."
            ),
            {
                "goal": self.state["goal"],
                "sources": self.store.manifest(),
                "findings": findings,
            },
            ReviewOutput,
        )

        expected = {finding["finding_id"] for finding in findings}
        received = {decision.finding_id for decision in result.decisions}
        if received != expected:
            raise ValueError("Checker omitted findings or invented IDs")

        self.state["reviews"].append({
            "revision": revision,
            **result.model_dump(mode="json"),
        })
        self.save_state()
        return result

    async def execute(self, goal):
        self.state["goal"] = goal
        self.save_state()
        self.logger.emit("run_started", goal=goal)

        accepted_ids = set()
        latest_review = None

        try:
            plan = await self.invoke(
                AgentRole.LEAD,
                "planning",
                (
                    "Create exactly two independent assignments: one for "
                    "evidence extraction and one for consistency analysis. "
                    "Both start simultaneously and independently read sources. "
                    "Neither receives the other assignment's output. "
                    "Do not instruct either agent to inspect, validate, or "
                    "cite the other assignment's extracted findings. "
                    "The evidence specialist extracts requested facts. "
                    "The consistency specialist independently compares source "
                    "statements and checks relevant policy or resolution "
                    "evidence. With one source, it checks for contradictions "
                    "or material gaps; it need not repeat extracted facts. "
                    "Preserve domain meanings: supported human languages "
                    "must not become programming languages. "
                    "Use only source IDs from the manifest. Give each "
                    "assignment a clear objective and acceptance criteria. "
                    "The assignments must jointly address the user's goal. "
                    "Preserve the user's goal in the plan."
                ),
                {"goal": goal, "sources": self.store.manifest()},
                Plan,
                tool_rounds=0,
            )

            known_sources = {
                item["source_id"] for item in self.store.manifest()
            }
            for assignment in plan.assignments:
                if not set(assignment.source_ids) <= known_sources:
                    raise ValueError("Plan contains unknown source IDs")

            self.state["plan"] = plan.model_dump(mode="json")
            self.save_state()

            # Both specialists share the same budget tracker and logger.
            self.logger.emit(
                "parallel_work_started",
                roles=[a.role.value for a in plan.assignments],
            )
            await asyncio.gather(*[
                self.analyze(assignment)
                for assignment in plan.assignments
            ])
            self.logger.emit("parallel_work_finished")

            for revision in range(self.settings.max_revision_rounds + 1):
                current_findings = self.findings()
                if not current_findings:
                    self.state["limitations"].append(
                        "Specialists produced no usable findings."
                    )
                    break

                try:
                    latest_review = await self.review(
                        current_findings, revision
                    )
                except Exception as exc:
                    self.state["limitations"].append(
                        f"Checker could not complete: {type(exc).__name__}."
                    )
                    break

                current_ids = {
                    finding["finding_id"] for finding in current_findings
                }
                accepted_ids.difference_update(current_ids)
                accepted_ids.update(
                    decision.finding_id
                    for decision in latest_review.decisions
                    if decision.verdict == "accept"
                )

                rejected = [
                    decision.model_dump(mode="json")
                    for decision in latest_review.decisions
                    if decision.verdict == "reject"
                ]
                gaps = latest_review.coverage_gaps

                if not rejected and not gaps:
                    break

                if revision >= self.settings.max_revision_rounds:
                    self.state["limitations"].append(
                        "Review issues remain after the revision limit."
                    )
                    break

                rejected_roles = {
                    decision["finding_id"].split(":")[0]
                    for decision in rejected
                }
                targets = [
                    assignment for assignment in plan.assignments
                    if gaps or assignment.role.value in rejected_roles
                ]

                before_revision = json.dumps(
                    self.findings(),
                    sort_keys=True,
                    ensure_ascii=False,
                )

                feedback = {
                    "rejected_findings": rejected,
                    "coverage_gaps": gaps,
                }
                self.logger.emit(
                    "revision_started",
                    revision=revision + 1,
                    roles=[a.role.value for a in targets],
                    feedback=feedback,
                )

                await asyncio.gather(*[
                    self.analyze(
                        assignment,
                        revision=revision + 1,
                        feedback=feedback,
                    )
                    for assignment in targets
                ])

                after_revision = json.dumps(
                    self.findings(),
                    sort_keys=True,
                    ensure_ascii=False,
                )

                if after_revision == before_revision:
                    self.logger.emit(
                        "review_skipped",
                        reason="Revision produced no changed findings",
                        revision=revision + 1,
                    )
                    self.state["limitations"].append(
                        "Revision produced no changed findings; "
                        "the previous checker decisions were preserved."
                    )
                    break

            current_findings = self.findings()
            accepted = [
                finding for finding in current_findings
                if finding["finding_id"] in accepted_ids
            ]
            unverified = [
                finding for finding in current_findings
                if finding["finding_id"] not in accepted_ids
            ]

            self.state["accepted_findings"] = accepted
            self.state["unverified_findings"] = unverified

            for role, task in self.state["tasks"].items():
                if task["status"] != "completed":
                    if task["revision"] > 0 and task["output"] is not None:
                        self.state["limitations"].append(
                            f"{role} revision {task['revision']} failed: "
                            f"{task['error']}. Earlier output was preserved."
                        )
                    else:
                        self.state["limitations"].append(
                            f"{role} initial analysis failed: "
                            f"{task['error']}."
                        )
                if task["output"]:
                    self.state["limitations"].extend(
                        task["output"]["unresolved_questions"]
                    )

            if latest_review:
                self.state["limitations"].extend(
                    latest_review.coverage_gaps
                )

            if unverified:
                self.state["limitations"].append(
                    f"{len(unverified)} findings are not accepted by "
                    "the checker; they are excluded from verified results."
                )

            if accepted:
                try:
                    selection = await self.invoke(
                        AgentRole.LEAD,
                        "final-selection",
                        (
                            "Order ALL supplied accepted finding IDs into "
                            "a useful report sequence. Do not omit, repeat, "
                            "or invent IDs."
                        ),
                        {
                            "goal": goal,
                            "accepted_findings": accepted,
                        },
                        FinalSelection,
                        tool_rounds=0,
                    )

                    by_id = {
                        finding["finding_id"]: finding
                        for finding in accepted
                    }
                    selected_ids = selection.ordered_finding_ids
                    unknown_ids = set(selected_ids) - set(by_id)

                    if unknown_ids:
                        raise ValueError(
                            "Final selection invented finding IDs"
                        )

                    selected_set = set(selected_ids)
                    omitted_ids = [
                        finding_id for finding_id in by_id
                        if finding_id not in selected_set
                    ]

                    if omitted_ids:
                        self.logger.emit(
                            "assembly_omissions_restored",
                            omitted_finding_ids=omitted_ids,
                            reason=(
                                "Preserve all accepted findings without "
                                "assuming omitted findings are duplicates"
                            ),
                        )

                    self.state["accepted_findings"] = [
                        by_id[finding_id]
                        for finding_id in selected_ids + omitted_ids
                    ]

                except Exception as exc:
                    self.state["limitations"].append(
                        "Lead assembly failed; preserved checked findings "
                        f"in deterministic order: {type(exc).__name__}."
                    )

        except Exception as exc:
            self.state["limitations"].append(
                f"Orchestration stopped: {type(exc).__name__}."
            )
            self.logger.emit(
                "orchestration_error",
                error_type=type(exc).__name__,
            )

        self.state["limitations"] = list(dict.fromkeys(
            self.state["limitations"]
        ))

        if self.state["accepted_findings"]:
            self.state["status"] = (
                "partial"
                if self.state["limitations"]
                or self.state["unverified_findings"]
                else "complete"
            )
        else:
            self.state["status"] = (
                "partial" if self.findings() else "failed"
            )
            # Preserve available work even if no checker review succeeded.
            self.state["unverified_findings"] = self.findings()

        self.state["budget"] = await self.budget.snapshot()
        self.save_state()
        self.write_report()

        self.logger.emit(
            "run_finished",
            status=self.state["status"],
            accepted_findings=len(self.state["accepted_findings"]),
            unverified_findings=len(self.state["unverified_findings"]),
            limitations=self.state["limitations"],
            budget=self.state["budget"],
        )

        return self.state

    def write_report(self):
        lines = [
            "# Document analysis",
            "",
            f"Status: **{self.state['status'].upper()}**",
            "",
            f"Goal: {self.state['goal']}",
            "",
            "## Checker-accepted findings",
            "",
        ]

        if not self.state["accepted_findings"]:
            lines.append("No findings received checker acceptance.")

        for finding in self.state["accepted_findings"]:
            lines.extend([
                f"### {finding['finding_id']}",
                "",
                finding["claim"],
                "",
            ])
            for citation in finding["citations"]:
                lines.append(
                    f"- {citation['source_id']}:"
                    f"{citation['start_line']}-{citation['end_line']}"
                )
                lines.extend(
                    f"> {line}" for line in citation["quote"].splitlines()
                )
                lines.append("")

            lines.extend([
                f"Reported uncertainty: {finding['uncertainty']}",
                "",
            ])

        lines.extend(["## Limitations", ""])
        lines.extend(
            f"- {item}" for item in self.state["limitations"]
        )
        if not self.state["limitations"]:
            lines.append("No execution or review gaps were reported.")

        lines.extend([
            "",
            "## Unverified or rejected findings",
            "",
        ])
        for finding in self.state["unverified_findings"]:
            lines.append(
                f"- {finding['finding_id']}: {finding['claim']}"
            )
        if not self.state["unverified_findings"]:
            lines.append("None.")

        (self.logger.run_dir / "report.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )


async def main(args):
    settings = load_settings()
    store = DocumentStore(args.input)

    try:
        answer = input(
            "Allow this multi-agent Groq run within configured budgets "
            "(account charges may apply)? [y/N]: "
        )
    except EOFError:
        print("No interactive input. No API requests made.")
        return

    if answer.strip().lower() != "y":
        print("Not approved. No API requests made.")
        return

    with EventLogger(
        "runs",
        secrets=(settings.groq_api_key.get_secret_value(),),
    ) as logger:
        logger.emit(
            "approval_decision",
            approved=True,
            scope="one multi-agent run including bounded revisions",
            model=settings.groq_model,
            limits=settings.model_dump(
                mode="json",
                exclude={"groq_api_key"},
            ),
        )

        budget = BudgetTracker(settings)
        client = ModelClient(settings, budget, logger, approved=True)

        try:
            orchestrator = Orchestrator(
                settings, store, logger, client, budget
            )
            result = await orchestrator.execute(args.goal)

            print("\nStatus:", result["status"].upper())
            print("Accepted findings:", len(result["accepted_findings"]))
            print("Unverified findings:", len(result["unverified_findings"]))
            for limitation in result["limitations"]:
                print("-", limitation)

            print("\nReport:", logger.run_dir / "report.md")
            print("State:", logger.run_dir / "state.json")
            print("Trace:", logger.run_dir)

        finally:
            await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--goal", required=True)
    arguments = parser.parse_args()
    asyncio.run(main(arguments))

