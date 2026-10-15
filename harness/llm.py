import asyncio
import json
import math
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from groq import AsyncGroq, APIConnectionError, APIStatusError

from harness.budget import BudgetExceeded, BudgetTracker
from harness.config import Settings
from harness.events import EventLogger
from harness.schemas import AgentRole


class ApprovalRequired(Exception):
    pass


class ModelClient:
    def __init__(
        self,
        settings: Settings,
        budget: BudgetTracker,
        logger: EventLogger,
        *,
        approved: bool = False,
    ):
        self.settings = settings
        self.budget = budget
        self.logger = logger
        self.approved = approved

        # Shared across all agents using this client.
        self._cooldown_until = 0.0

        self.client = AsyncGroq(
            api_key=settings.groq_api_key.get_secret_value(),
            max_retries=0,
            timeout=settings.request_timeout_seconds,
        )

    async def close(self):
        await self.client.close()

    @staticmethod
    def prompt_allowance(payload: dict) -> int:
        serialized = json.dumps(payload, ensure_ascii=False)
        return len(serialized.encode("utf-8")) + 512

    @staticmethod
    def retryable(exc: Exception) -> bool:
        if isinstance(exc, (APIConnectionError, TimeoutError)):
            return True
        if isinstance(exc, APIStatusError):
            return (
                exc.status_code in {408, 409, 429}
                or exc.status_code >= 500
            )
        return False

    @staticmethod
    def retry_after_seconds(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        if response is None:
            return None

        value = response.headers.get("retry-after")
        if not value:
            return None

        try:
            seconds = float(value)
        except ValueError:
            try:
                when = parsedate_to_datetime(value)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                seconds = (
                    when - datetime.now(timezone.utc)
                ).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return None

        if not math.isfinite(seconds):
            return None
        return max(0.0, seconds)

    @staticmethod
    def rate_headers(exc: Exception) -> dict:
        response = getattr(exc, "response", None)
        if response is None:
            return {}

        names = (
            "retry-after",
            "x-ratelimit-limit-requests",
            "x-ratelimit-limit-tokens",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
        )
        return {
            name: response.headers[name]
            for name in names
            if name in response.headers
        }

    async def _wait_for_cooldown(self, role, task_id):
        while True:
            delay = self._cooldown_until - time.monotonic()
            if delay <= 0:
                return

            remaining = await self.budget.remaining_seconds(role)
            if delay >= remaining:
                raise BudgetExceeded(
                    "Provider cooldown exceeds remaining deadline"
                )

            self.logger.emit(
                "provider_cooldown",
                agent=role,
                task_id=task_id,
                delay_seconds=delay,
            )

            # Recheck in case another in-flight request extends cooldown.
            await asyncio.sleep(min(delay, 30.0))

    async def complete(
        self,
        role: AgentRole,
        messages: list[dict],
        *,
        task_id: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        tool_choice: str | None = None,
    ):
        if not self.approved:
            self.logger.emit(
                "approval_required",
                agent=role,
                task_id=task_id,
                action="groq_inference",
            )
            raise ApprovalRequired("Groq inference requires approval")

        if tools and response_format:
            raise ValueError(
                "Use separate requests for tools and structured outputs"
            )

        payload = {
            "model": self.settings.groq_model,
            "messages": messages,
            "reasoning_effort": "low",
            "max_completion_tokens": self.settings.max_completion_tokens,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            if not tools:
                raise ValueError("tool_choice requires tools")
            if tool_choice not in {"auto", "required", "none"}:
                raise ValueError("Unsupported tool_choice")
            payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format

        input_allowance = self.prompt_allowance(payload)

        for attempt in range(self.settings.max_retries + 1):
            try:
                # Wait before reserving tokens for the next attempt.
                await self._wait_for_cooldown(role, task_id)
                reservation = await self.budget.reserve_model(
                    role,
                    input_allowance,
                    self.settings.max_completion_tokens,
                )
            except BudgetExceeded as exc:
                self.logger.emit(
                    "budget_stop",
                    agent=role,
                    task_id=task_id,
                    reason=str(exc),
                )
                raise

            started = time.monotonic()
            actual_tokens = None
            dispatched = False
            rate_rejected = False
            response = None
            failure = None
            retry_delay = None

            try:
                remaining = await self.budget.remaining_seconds(role)
                if remaining <= 0:
                    raise BudgetExceeded("Deadline reached before dispatch")

                timeout = min(
                    remaining,
                    self.settings.request_timeout_seconds,
                )

                self.logger.emit(
                    "model_request",
                    agent=role,
                    task_id=task_id,
                    attempt=attempt + 1,
                    reservation_id=reservation.reservation_id,
                    reserved_tokens=reservation.tokens,
                    timeout_seconds=timeout,
                    request=payload,
                )

                dispatched = True
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(**payload),
                    timeout=timeout,
                )

                if response.usage is not None:
                    actual_tokens = response.usage.total_tokens

                choices = [
                    {
                        "finish_reason": choice.finish_reason,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                            "tool_calls": [
                                call.model_dump(mode="json")
                                for call in (choice.message.tool_calls or [])
                            ],
                        },
                    }
                    for choice in response.choices
                ]

                self.logger.emit(
                    "model_response",
                    agent=role,
                    task_id=task_id,
                    attempt=attempt + 1,
                    response_id=response.id,
                    model=response.model,
                    choices=choices,
                    usage=(
                        response.usage.model_dump(mode="json")
                        if response.usage else None
                    ),
                    duration_seconds=time.monotonic() - started,
                )

            except asyncio.CancelledError:
                self.logger.emit(
                    "model_cancelled",
                    agent=role,
                    task_id=task_id,
                    attempt=attempt + 1,
                    duration_seconds=time.monotonic() - started,
                )
                raise

            except Exception as exc:
                failure = exc
                rate_rejected = (
                    isinstance(exc, APIStatusError)
                    and exc.status_code == 429
                )

                if rate_rejected:
                    provider_delay = self.retry_after_seconds(exc)
                    fallback = min(60.0, 5.0 * (2 ** attempt))
                    retry_delay = (
                        provider_delay
                        if provider_delay is not None
                        else fallback
                    ) + random.uniform(0.1, 0.5)

                    # No await between reading and updating shared state.
                    self._cooldown_until = max(
                        self._cooldown_until,
                        time.monotonic() + retry_delay,
                    )

                self.logger.emit(
                    "model_error",
                    agent=role,
                    task_id=task_id,
                    attempt=attempt + 1,
                    error_type=type(exc).__name__,
                    status_code=getattr(exc, "status_code", None),
                    rate_limit_headers=self.rate_headers(exc),
                    provider_error=(
                        {
                            "message": exc.body.get(
                                "error", exc.body
                            ).get("message"),
                            "code": exc.body.get(
                                "error", exc.body
                            ).get("code"),
                        }
                        if isinstance(exc, APIStatusError)
                        and isinstance(exc.body, dict)
                        and isinstance(
                            exc.body.get("error", exc.body), dict
                        )
                        else None
                    ),
                    duration_seconds=time.monotonic() - started,
                )

            finally:
                if not dispatched:
                    charge = 0
                    basis = "not_dispatched"
                elif rate_rejected:
                    charge = 0
                    basis = "http_429_rejection_policy"
                else:
                    charge = actual_tokens
                    basis = (
                        "provider_reported"
                        if actual_tokens is not None
                        else "unknown_usage_full_reservation"
                    )

                settlement = await self.budget.settle_model(
                    reservation,
                    actual_tokens=charge,
                )

                # A released 429 reservation is not a provider usage report.
                settlement["usage_known"] = (
                    actual_tokens is not None or not dispatched
                )

                self.logger.emit(
                    "token_settlement",
                    agent=role,
                    task_id=task_id,
                    reservation_id=reservation.reservation_id,
                    accounting_basis=basis,
                    **settlement,
                )

            if (
                settlement["reservation_exceeded"]
                or settlement["budget_exceeded"]
            ):
                self.logger.emit(
                    "budget_stop",
                    agent=role,
                    task_id=task_id,
                    reason="Token reservation or budget exceeded",
                )
                raise BudgetExceeded("Token reservation or budget exceeded")

            if failure is None:
                return response

            if (
                not self.retryable(failure)
                or attempt >= self.settings.max_retries
            ):
                raise failure

            if retry_delay is None:
                retry_delay = (2 ** attempt) + random.uniform(0, 0.25)

            remaining = await self.budget.remaining_seconds(role)
            if retry_delay >= remaining:
                self.logger.emit(
                    "budget_stop",
                    agent=role,
                    task_id=task_id,
                    reason="Retry wait exceeds remaining deadline",
                )
                raise BudgetExceeded(
                    "Insufficient time for retry"
                ) from failure

            self.logger.emit(
                "model_retry",
                agent=role,
                task_id=task_id,
                next_attempt=attempt + 2,
                delay_seconds=retry_delay,
                shared_cooldown=rate_rejected,
            )

            if not rate_rejected:
                await asyncio.sleep(retry_delay)
            # For HTTP 429, the next iteration waits on shared cooldown.

