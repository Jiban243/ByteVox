import asyncio

import pytest

from harness.budget import BudgetExceeded, BudgetTracker
from harness.config import Settings
from harness.schemas import AgentRole


@pytest.mark.asyncio
async def test_concurrent_agents_cannot_reserve_the_same_budget():
    settings = Settings(
        groq_api_key="test-only",
        groq_model="test-only",
        run_max_tokens=100,
        agent_max_tokens=100,
        max_completion_tokens=20,
    )
    tracker = BudgetTracker(settings)

    results = await asyncio.gather(
        tracker.reserve_model(AgentRole.EVIDENCE, 40, 20),
        tracker.reserve_model(AgentRole.CONSISTENCY, 40, 20),
        return_exceptions=True,
    )

    accepted = [
        result for result in results
        if not isinstance(result, BaseException)
    ]
    rejected = [
        result for result in results
        if isinstance(result, BudgetExceeded)
    ]

    assert len(accepted) == 1
    assert len(rejected) == 1

    # Reconcile the successful request with its actual usage.
    await tracker.settle_model(accepted[0], actual_tokens=30)

    snapshot = await tracker.snapshot()
    assert snapshot["run"]["charged_tokens"] == 30
    assert snapshot["run"]["reserved_tokens"] == 0
    assert snapshot["pending_reservations"] == 0


@pytest.mark.asyncio
async def test_missing_usage_retains_charge_and_prevents_double_settlement():
    settings = Settings(
        groq_api_key="test-only",
        groq_model="test-only",
    )
    tracker = BudgetTracker(settings)

    reservation = await tracker.reserve_model(
        AgentRole.EVIDENCE, 100, 200
    )
    outcome = await tracker.settle_model(
        reservation, actual_tokens=None
    )

    assert outcome["charged_tokens"] == 300
    assert outcome["usage_known"] is False

    with pytest.raises(ValueError):
        await tracker.settle_model(reservation, actual_tokens=50)


@pytest.mark.asyncio
async def test_tool_attempts_obey_agent_step_limit():
    settings = Settings(
        groq_api_key="test-only",
        groq_model="test-only",
        agent_max_steps=1,
    )
    tracker = BudgetTracker(settings)

    await tracker.begin_tool(AgentRole.EVIDENCE)

    with pytest.raises(BudgetExceeded):
        await tracker.begin_tool(AgentRole.EVIDENCE)

