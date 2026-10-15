import asyncio

from harness.budget import BudgetTracker
from harness.config import Settings
from harness.events import EventLogger
from harness.llm import ModelClient
from harness.orchestrator import Orchestrator
from harness.tools import DocumentStore


async def main():
    # No real credentials are loaded.
    # The 100-token budget cannot admit the planning request.
    settings = Settings(
        groq_api_key="unused-offline-demo",
        groq_model="openai/gpt-oss-120b",
        run_max_tokens=100,
        agent_max_tokens=100,
        max_completion_tokens=20,
    )

    store = DocumentStore("data/smoke")

    with EventLogger("runs") as logger:
        logger.emit(
            "demonstration",
            scenario="planning_request_exceeds_token_budget",
            deliberately_small_budget=True,
            expected_api_requests=0,
        )

        budget = BudgetTracker(settings)
        client = ModelClient(
            settings,
            budget,
            logger,
            approved=True,
        )

        # Fail closed if any request unexpectedly reaches the transport.
        # The normal budget guard should stop execution before this.
        network_attempts = 0

        async def forbid_network(*args, **kwargs):
            nonlocal network_attempts
            network_attempts += 1
            raise RuntimeError("Network disabled for offline demonstration")

        client.client.chat.completions.create = forbid_network

        try:
            orchestrator = Orchestrator(
                settings, store, logger, client, budget
            )

            result = await orchestrator.execute(
                "Identify launch-date and language disagreements."
            )

            import json

            events = [
                json.loads(line)
                for line in logger.path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            stops = [
                event
                for event in events
                if event["event"] == "budget_stop"
            ]

            assert network_attempts == 0, "Unexpected transport attempt"
            assert stops, "No budget-stop event recorded"
            assert result["status"] == "failed"
            assert result["budget"]["pending_reservations"] == 0

            print("\nLimit demonstration passed.")
            print("Configured run token budget: 100")
            print("API requests dispatched: 0")
            print("Reason:", stops[0]["data"]["reason"])
            print("Status:", result["status"].upper())
            print("Report:", logger.run_dir / "report.md")
            print("Trace:", logger.run_dir)

        finally:
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())

