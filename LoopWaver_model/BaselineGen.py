from llms.llms import async_call_llm
import json
import time
import logging
import asyncio
from json_repair import repair_json
import re

class BaselineGen:

    """Baseline generator: direct single-stage generation."""
    @staticmethod
    async def async_generate(model, example, semaphore):
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                time_start = time.time()
                async with semaphore:
                    response = await async_call_llm(model, example["prompt"])
                    time_end = time.time()

                    logging.info(f"Time taken: {time_end - time_start} seconds")

                    logging.info(f"Response: {response}")

                    response = str(response)
                    example["response"] = response
                    example["output_blocks"] = response.split('#*#')
                    example["word_count"] = len(response.split())
                    example["time_taken"] = time_end - time_start
                    example["final_text"] = response

                return example

            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    logging.error(f"Failed to process example after {max_retries} attempts: {e}")
                    raise
                logging.warning(f"Attempt {retry_count} failed: {e}. Retrying...")
                await asyncio.sleep(1 * retry_count)
