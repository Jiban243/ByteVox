from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
PositiveInt = Annotated[int, Field(gt=0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentRole(str, Enum):
    LEAD = "lead"
    EVIDENCE = "evidence"
    CONSISTENCY = "consistency"
    CHECKER = "checker"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    BUDGET_EXCEEDED = "budget_exceeded"


class RunStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


# These are the two roles that receive analysis assignments.
AnalysisRole = Literal[
    AgentRole.EVIDENCE,
    AgentRole.CONSISTENCY,
]


class Assignment(StrictModel):
    task_id: NonEmptyText
    role: AnalysisRole
    objective: NonEmptyText
    source_ids: list[NonEmptyText] = Field(min_length=1)
    acceptance_criteria: list[NonEmptyText] = Field(min_length=1)


class Plan(StrictModel):
    goal: NonEmptyText
    assignments: list[Assignment] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_assignments(self):
        task_ids = [item.task_id for item in self.assignments]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Task IDs must be unique")

        roles = {item.role for item in self.assignments}
        expected = {AgentRole.EVIDENCE, AgentRole.CONSISTENCY}
        if roles != expected:
            raise ValueError(
                "Plan needs one evidence and one consistency assignment"
            )

        return self


class Citation(StrictModel):
    source_id: NonEmptyText
    start_line: PositiveInt
    end_line: PositiveInt
    quote: NonEmptyText

    @model_validator(mode="after")
    def validate_line_range(self):
        if self.end_line < self.start_line:
            raise ValueError("end_line cannot precede start_line")
        return self


class Finding(StrictModel):
    finding_id: NonEmptyText
    claim: NonEmptyText
    citations: list[Citation] = Field(min_length=1)
    uncertainty: str


class AnalysisOutput(StrictModel):
    """Model-produced content, without runtime-controlled status fields."""

    findings: list[Finding]
    unresolved_questions: list[NonEmptyText]

    @model_validator(mode="after")
    def validate_finding_ids(self):
        ids = [finding.finding_id for finding in self.findings]
        if len(ids) != len(set(ids)):
            raise ValueError("Finding IDs must be unique within an output")
        return self


class Usage(StrictModel):
    """Populated by the harness from execution and provider usage."""

    model_requests: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0, ge=0)

    # A failed request may not return token usage.
    usage_complete: bool = True


class TaskResult(StrictModel):
    """Runtime envelope. Agents do not declare their own execution status."""

    task_id: NonEmptyText
    role: AnalysisRole
    status: TaskStatus
    output: AnalysisOutput | None
    error: str | None
    usage: Usage

    @model_validator(mode="after")
    def validate_terminal_result(self):
        if self.status in {TaskStatus.PENDING, TaskStatus.RUNNING}:
            raise ValueError("TaskResult requires a terminal status")

        if self.status == TaskStatus.COMPLETED:
            if self.output is None:
                raise ValueError("Completed tasks require an output")
            if self.error is not None:
                raise ValueError("Completed tasks cannot have an error")
        elif not self.error or not self.error.strip():
            raise ValueError("Unsuccessful tasks require an error reason")

        return self


class ReviewDecision(StrictModel):
    finding_id: NonEmptyText
    verdict: Literal["accept", "reject"]
    reason: NonEmptyText
    requested_change: str | None

    @model_validator(mode="after")
    def validate_rejection(self):
        if self.verdict == "reject":
            if not self.requested_change or not self.requested_change.strip():
                raise ValueError("Rejected findings need a requested change")
        return self


class ReviewOutput(StrictModel):
    decisions: list[ReviewDecision]
    coverage_gaps: list[NonEmptyText]

    @model_validator(mode="after")
    def validate_decision_ids(self):
        ids = [decision.finding_id for decision in self.decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("A finding cannot have multiple review decisions")
        return self


class FinalSelection(StrictModel):
    """Lead selects checked findings; it cannot add uncited report claims."""

    ordered_finding_ids: list[NonEmptyText]

    @model_validator(mode="after")
    def validate_selection(self):
        ids = self.ordered_finding_ids
        if len(ids) != len(set(ids)):
            raise ValueError("Final selection cannot repeat finding IDs")
        return self
    
