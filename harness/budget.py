import asyncio
import time
from dataclasses import dataclass

from harness.config import Settings
from harness.schemas import AgentRole


class BudgetExceeded(Exception):
    pass


@dataclass
class Counters:
    steps: int = 0
    charged_tokens: int = 0
    reserved_tokens: int = 0


@dataclass(frozen=True)
class Reservation:
    reservation_id: int
    role: AgentRole
    tokens: int


class BudgetTracker:
    """
    One tracker per run.

    Token charges represent reported usage when available.
    If a dispatched request returns no usage, retain its full reservation
    as a conservative budget charge, not as measured token usage.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._clock = time.monotonic
        self._run_started = self._clock()

        self._agent_started: dict[AgentRole, float] = {}
        self._agents = {role: Counters() for role in AgentRole}
        self._run = Counters()

        self._lock = asyncio.Lock()
        self._next_id = 0
        self._reservations: dict[int, Reservation] = {}

    def _remaining_seconds(self, role: AgentRole) -> float:
        now = self._clock()
        run_remaining = (
            self.settings.run_timeout_seconds
            - (now - self._run_started)
        )

        started = self._agent_started.get(role)
        agent_remaining = self.settings.agent_timeout_seconds
        if started is not None:
            agent_remaining -= now - started

        return max(0.0, min(run_remaining, agent_remaining))

    async def remaining_seconds(self, role: AgentRole) -> float:
        async with self._lock:
            return self._remaining_seconds(role)

    def _check_action(self, role: AgentRole) -> None:
        if self._remaining_seconds(role) <= 0:
            raise BudgetExceeded(f"{role.value}: deadline reached")

        if self._run.steps >= self.settings.run_max_steps:
            raise BudgetExceeded("Run step budget exhausted")

        if self._agents[role].steps >= self.settings.agent_max_steps:
            raise BudgetExceeded(f"{role.value}: step budget exhausted")

    def _record_step(self, role: AgentRole) -> None:
        self._agent_started.setdefault(role, self._clock())
        self._run.steps += 1
        self._agents[role].steps += 1

    async def begin_tool(self, role: AgentRole) -> float:
        """Charge one tool attempt and return its remaining time allowance."""
        async with self._lock:
            self._check_action(role)
            self._record_step(role)
            return self._remaining_seconds(role)

    async def reserve_model(
        self,
        role: AgentRole,
        prompt_token_allowance: int,
        completion_token_allowance: int,
    ) -> Reservation:
        """
        Reserve input + maximum completion allowance before dispatch.

        The model client must calculate the input allowance and use the
        same completion allowance in the API request.
        """
        if prompt_token_allowance < 0 or completion_token_allowance <= 0:
            raise ValueError("Invalid token allowance")

        requested = prompt_token_allowance + completion_token_allowance

        async with self._lock:
            self._check_action(role)
            agent = self._agents[role]

            run_total = (
                self._run.charged_tokens
                + self._run.reserved_tokens
                + requested
            )
            agent_total = (
                agent.charged_tokens
                + agent.reserved_tokens
                + requested
            )

            if run_total > self.settings.run_max_tokens:
                raise BudgetExceeded(
                    "Insufficient run token budget for this request"
                )

            if agent_total > self.settings.agent_max_tokens:
                raise BudgetExceeded(
                    f"{role.value}: insufficient token budget"
                )

            self._record_step(role)
            self._run.reserved_tokens += requested
            agent.reserved_tokens += requested

            self._next_id += 1
            reservation = Reservation(self._next_id, role, requested)
            self._reservations[reservation.reservation_id] = reservation
            return reservation

    async def settle_model(
        self,
        reservation: Reservation,
        actual_tokens: int | None,
    ) -> dict:
        """
        Settle exactly once, including after timeout or cancellation.

        None means usage is unavailable: charge the reserved allowance.
        A measured overrun is recorded honestly and blocks further model
        requests once the budget is exhausted.
        """
        if actual_tokens is not None and actual_tokens < 0:
            raise ValueError("Token usage cannot be negative")

        async with self._lock:
            stored = self._reservations.get(reservation.reservation_id)
            if stored != reservation:
                raise ValueError("Unknown or already-settled reservation")

            del self._reservations[reservation.reservation_id]
            agent = self._agents[reservation.role]

            charged = (
                reservation.tokens
                if actual_tokens is None
                else actual_tokens
            )

            self._run.reserved_tokens -= reservation.tokens
            agent.reserved_tokens -= reservation.tokens

            self._run.charged_tokens += charged
            agent.charged_tokens += charged

            return {
                "charged_tokens": charged,
                "usage_known": actual_tokens is not None,
                "reservation_exceeded": charged > reservation.tokens,
                "budget_exceeded": (
                    self._run.charged_tokens
                    + self._run.reserved_tokens
                    > self.settings.run_max_tokens
                    or agent.charged_tokens
                    + agent.reserved_tokens
                    > self.settings.agent_max_tokens
                ),
            }

    async def snapshot(self) -> dict:
        async with self._lock:
            def counters(value: Counters) -> dict:
                return {
                    "steps": value.steps,
                    "charged_tokens": value.charged_tokens,
                    "reserved_tokens": value.reserved_tokens,
                }

            return {
                "elapsed_seconds": self._clock() - self._run_started,
                "run": counters(self._run),
                "agents": {
                    role.value: counters(value)
                    for role, value in self._agents.items()
                },
                "pending_reservations": len(self._reservations),
            }

