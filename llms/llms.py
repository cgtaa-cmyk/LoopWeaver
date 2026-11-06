import logging
import os
from openai import OpenAI, AsyncOpenAI
from transformers import pipeline
import torch
import asyncio
import httpx
from contextlib import asynccontextmanager
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import openai
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_clients = {}

@asynccontextmanager
async def get_client(base_url, api_key):
    """Get or create an AsyncOpenAI client."""
    client_key = f"{base_url}:{api_key}"
    if client_key not in _clients:
        long_timeout_async_client = httpx.AsyncClient(timeout=900, trust_env=False)
        _clients[client_key] = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=long_timeout_async_client
        )
    try:
        yield _clients[client_key]
    finally:
        pass

def create_retry_decorator(max_retries=8, min_wait=1, max_wait=60):
    return retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.APIError,
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
        )),
        before_sleep=lambda retry_state: logger.warning(
            f"API call failed with {retry_state.outcome.exception()}, "
            f"retrying in {retry_state.next_action.sleep} seconds..."
        )
    )

@create_retry_decorator()
async def _make_api_call(client, **kwargs):
    """Helper function to make API calls with retry logic"""
    return await client.chat.completions.create(**kwargs)

def call_llm(model, prompt):
    return asyncio.run(async_call_llm(model, prompt))


def _is_planning_prompt(prompt):
    if "your task is to" in prompt:
        return True
    if "revise this floor plan" in prompt or "revise this block plan" in prompt:
        return True
    return "Create a precise" in prompt and "plan" in prompt.lower()


async def async_call_llm(model, prompt):
    try:
        if model in ["gpt-4o-mini"]:
            api_key = os.environ.get("GPT4O_API_KEY", "")
            base_url = os.environ.get("GPT4O_BASE_URL", "")
            plan_max = int(os.environ.get("GPT4O_PLAN_MAX_TOKENS", "20000"))
            gen_max = int(os.environ.get("GPT4O_GEN_MAX_TOKENS", "4000"))
            max_tokens = plan_max if _is_planning_prompt(prompt) else gen_max
            temperature = float(os.environ.get("GPT4O_TEMPERATURE", "0.7"))
            async with get_client(base_url, api_key) as client:
                completion = await _make_api_call(
                    client,
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if completion is None:
                    logging.error(f"{model}: API returned None response")
                    return ""

                if not hasattr(completion, 'choices') or completion.choices is None:
                    logging.error(f"{model}: response has no choices field")
                    return ""

                if len(completion.choices) == 0:
                    logging.error(f"{model}: choices list is empty")
                    return ""

                if completion.choices[0] is None:
                    logging.error(f"{model}: first choice is None")
                    return ""

                if not hasattr(completion.choices[0], 'message') or completion.choices[0].message is None:
                    logging.error(f"{model}: choice has no message field")
                    return ""

                if not hasattr(completion.choices[0].message, 'content') or completion.choices[0].message.content is None:
                    logging.error(f"{model}: message has no content field")
                    return ""

                content = completion.choices[0].message.content
                if not content or content.strip() == "":
                    logging.warning(f"{model}: returned content is empty")
                    return ""

                logging.info(f"{model} API Call Successful")
                return content

        elif model in ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]:
            async with get_client("http://0.0.0.0:8004/v1", "123456") as client:
                model_dict = {
                    "Qwen2.5-0.5B-Instruct": "/path/to/models/Qwen2.5-0.5B-Instruct",
                    "Qwen2.5-1.5B-Instruct": "/path/to/models/Qwen2.5-1.5B-Instruct",
                    "Qwen2.5-7B-Instruct": "/path/to/models/Qwen2.5-7B-Instruct",
                    "Llama-3.1-8B-Instruct": "/path/to/models/Llama-3.1-8B-Instruct"
                }
                plan_max = int(os.environ.get("LOOPWEAVER_PLAN_MAX_TOKENS", "12000"))
                if _is_planning_prompt(prompt):
                    max_tokens = plan_max
                elif os.environ.get("LOOPWEAVER_BATCH_GENERATION", "0") == "1":
                    max_tokens = int(os.environ.get("LOOPWEAVER_BATCH_MAX_TOKENS", "5000"))
                else:
                    max_tokens = 1200

                served_model = os.environ.get("LOOPWEAVER_API_MODEL_ID", model_dict[model])
                response = await _make_api_call(
                    client,
                    model=served_model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                )
                logging.info(f"{model} API Call Successful")
                return response.choices[0].message.content

    except Exception as e:
        logging.error(f"API Call Failed after retries: {e}")
        return ""
