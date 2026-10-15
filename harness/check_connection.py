import asyncio

from harness.budget import BudgetTracker
from harness.config import load_settings
from harness.events import EventLogger
from harness.llm import ModelClient
from harness.schemas import AgentRole


async def main():
    settings = load_settings()

    try:
        answer = input(
            "Allow this Groq inference check "
            "(up to 3 attempts; account charges may apply)? [y/N]: "
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
            scope="single inference check with bounded retries",
            model=settings.groq_model,
        )

        if not approved:
            print("No API request made.")
            return

        budget = BudgetTracker(settings)
        client = ModelClient(
            settings, budget, logger, approved=True
        )

        try:
            response = await client.complete(
                AgentRole.LEAD,
                [{
                    "role": "user",
                    "content": "Reply with exactly: Integrated client ready",
                }],
                task_id="connection-check",
            )
            print("Response:", response.choices[0].message.content)
            print("Budget:", await budget.snapshot())
        finally:
            await client.close()
            print("Trace:", logger.run_dir)


if __name__ == "__main__":
    asyncio.run(main())

