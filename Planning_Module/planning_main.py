import asyncio
import logging
import json
import os
import argparse
from tqdm.asyncio import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LoopWaver_model.Agents.PlanningAgent import PlanningAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

async def process_example(model, example, semaphore, checkpoint_dir):
    """Process the plan generation flow for a single example, including checkpoint and error handling."""

    example_id = example.get('id', f"{example['type']}_{hash(example['prompt'])}")
    checkpoint_file = os.path.join(checkpoint_dir, f'plan_checkpoint_{example_id}.json')

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                logging.info(f"Loading checkpoint for {example_id}")
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading checkpoint for {example_id}: {e}")

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            logging.info(f"Creating plan for example {example_id}")
            example = await PlanningAgent.async_create_hierarchy(model, example, semaphore)

            try:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(example, f)
            except Exception as e:
                logging.warning(f"Failed to save checkpoint: {e}")

            return example

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                logging.error(f"Failed to process example {example_id} after {max_retries} attempts: {e}")
                example['error'] = str(e)
                return example
            logging.warning(f"Attempt {retry_count} failed for example {example_id}: {e}. Retrying...")
            await asyncio.sleep(1 * retry_count)

async def main():
    """Main function: process all examples, supports command-line arguments."""

    parser = argparse.ArgumentParser(description="Generate plans using PlanningAgent.")
    parser.add_argument("--model", type=str, default="Qwen2.5-0.5B-Instruct",
                      help="Specify the model name (default: Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--output_dir", type=str, default="planning_results",
                      help="Specify the output directory (default: planning_results)")
    parser.add_argument("--max_concurrent", type=int, default=5,
                      help="Maximum number of concurrent tasks (default: 5)")
    parser.add_argument("--examples_file", type=str, default="../datasets/Dataset_short.json",
                      help="Path to a JSON file containing examples")
    parser.add_argument("--sample_count", type=int, default=400,
                      help="Number of examples to sample from the dataset (default: 400)")

    args = parser.parse_args()

    output_dir = args.output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(args.max_concurrent)

    if args.examples_file and os.path.exists(args.examples_file):
        with open(args.examples_file, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        logging.info(f"Loaded {len(examples)} examples from {args.examples_file}")

        import random
        if args.sample_count and args.sample_count < len(examples):
            examples = random.sample(examples, args.sample_count)
            logging.info(f"Sampled {len(examples)} examples for testing")
    else:
        logging.error("Examples file not found or invalid.")
        return

    for i, example in enumerate(examples):
        example['id'] = f"{example['type']}_{i}"

    tasks = [process_example(args.model, example, semaphore, checkpoint_dir) for example in examples]
    results = await tqdm.gather(*tasks, desc="Generating plans")

    has_errors = False
    for result in results:
        if 'error' in result:
            has_errors = True
            logging.error(f"Example {result.get('id', 'unknown')} failed: {result['error']}")

    final_output_file = os.path.join(output_dir, "planning_results.json")
    logging.info(f"Writing final results to {final_output_file}")
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logging.info("=" * 50)
    logging.info("PLANNING RESULTS SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Total examples processed: {len(results)}")
    logging.info(f"Failed examples: {sum(1 for r in results if 'error' in r)}")
    logging.info("=" * 50)

    if has_errors:
        logging.warning("Some examples failed to process. Check logs for details.")

    return results

if __name__ == "__main__":
    try:
        results = asyncio.run(main())
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")
        raise
