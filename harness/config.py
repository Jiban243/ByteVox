from pathlib import Path
from typing import Annotated

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
import os


PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    groq_api_key: SecretStr = Field(repr=False)
    groq_model: str = Field(min_length=1)

    run_timeout_seconds: PositiveFloat = 180
    agent_timeout_seconds: PositiveFloat = 90
    request_timeout_seconds: PositiveFloat = 30

    run_max_steps: PositiveInt = 50
    agent_max_steps: PositiveInt = 15

    run_max_tokens: PositiveInt = 30_000
    agent_max_tokens: PositiveInt = 10_000
    max_completion_tokens: PositiveInt = 2_000

    max_retries: Annotated[int, Field(ge=0, le=2)] = 2
    max_revision_rounds: Annotated[int, Field(ge=0, le=1)] = 1

    @model_validator(mode="after")
    def validate_limits(self):
        if not self.groq_api_key.get_secret_value().strip():
            raise ValueError("GROQ_API_KEY must not be blank")

        if not self.groq_model.strip():
            raise ValueError("GROQ_MODEL must not be blank")

        if self.agent_timeout_seconds > self.run_timeout_seconds:
            raise ValueError("Agent timeout cannot exceed run timeout")

        if self.request_timeout_seconds > self.agent_timeout_seconds:
            raise ValueError("Request timeout cannot exceed agent timeout")

        if self.agent_max_steps > self.run_max_steps:
            raise ValueError("Agent step budget cannot exceed run budget")

        if self.agent_max_tokens > self.run_max_tokens:
            raise ValueError("Agent token budget cannot exceed run budget")

        if self.max_completion_tokens >= self.agent_max_tokens:
            raise ValueError(
                "Completion allowance must leave room for input tokens"
            )

        return self


def load_settings(env_file: str | Path = ".env") -> Settings:
    # Explicit environment variables take precedence over .env values.
    values = {
        **dotenv_values(env_file),
        **os.environ,
    }

    arguments = {}
    for field_name in Settings.model_fields:
        environment_name = field_name.upper()
        if environment_name in values:
            arguments[field_name] = values[environment_name]

    return Settings.model_validate(arguments)


if __name__ == "__main__":
    from pydantic import ValidationError

    try:
        settings = load_settings()
    except ValidationError as exc:
        # Do not print raw validation errors: they can contain input values.
        print("Configuration invalid:")
        for error in exc.errors(include_input=False, include_url=False):
            field = ".".join(str(part) for part in error["loc"]) or "settings"
            print(f"- {field}: {error['msg']}")
        raise SystemExit(1)

    print("Configuration valid")
    print(settings.model_dump_json(
        indent=2,
        exclude={"groq_api_key"},
    ))

