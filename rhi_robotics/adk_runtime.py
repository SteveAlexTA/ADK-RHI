import asyncio
import os

from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


def _is_retryable_provider_error(exception: BaseException) -> bool:
    if type(exception).__name__ in {
        "APIConnectionError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
    }:
        return True
    status_code = getattr(exception, "status_code", None)
    return status_code in {408, 409, 429, 500, 502, 503, 504}


@retry(
    retry=retry_if_exception(_is_retryable_provider_error),
    wait=wait_random_exponential(min=10, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)
async def run_adk_agent(
    agent_name: str,
    model_name: str,
    instruction: str,
    user_prompt: str,
    temperature: float,
    output_schema: type[BaseModel] | None = None,
) -> str:
    """Run a single-turn ADK agent using Gemini and return its final text."""
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it to the .env file.")

    app_name = "rhi_robotics"
    user_id = "rhi_robotics"
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
    )
    agent = LlmAgent(
        name=agent_name,
        model=model_name,
        instruction=instruction,
        output_schema=output_schema,
        generate_content_config=types.GenerateContentConfig(temperature=temperature),
    )
    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
    )
    message = types.Content(
        role="user",
        parts=[types.Part(text=user_prompt)],
    )
    final_response = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=message,
    ):
        if event.is_final_response():
            final_response = "".join(
                part.text
                for part in (event.content.parts if event.content else [])
                if part.text
            )

    if final_response:
        return final_response
    raise RuntimeError(f"The ADK agent {agent_name} returned no final text response.")


def run_agent(
    agent_name: str,
    model_name: str,
    instruction: str,
    user_prompt: str,
    temperature: float,
    output_schema: type[BaseModel] | None = None,
) -> str:
    """Run an ADK agent from the synchronous RHI orchestration loop."""
    return asyncio.run(
        run_adk_agent(
            agent_name=agent_name,
            model_name=model_name,
            instruction=instruction,
            user_prompt=user_prompt,
            temperature=temperature,
            output_schema=output_schema,
        )
    )
