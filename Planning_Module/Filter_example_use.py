import asyncio
import logging
import json
import os
import argparse
from tqdm.asyncio import tqdm


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LoopWaver_model.Agents.PlanningAgent import PlanningAgent
from LoopWaver_model.Agents.FilterAgent import FilterAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

async def process_example(model, example, semaphore, checkpoint_dir):
    """Process the plan generation and filtering flow for a single example, including checkpoint and error handling."""

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
            logging.info(f"Step 1: Creating plan for example {example_id}")
            example = await PlanningAgent.async_create_hierarchy(model, example, semaphore)

            try:
                with open(checkpoint_file + '.plan', 'w', encoding='utf-8') as f:
                    json.dump(example, f)
            except Exception as e:
                logging.warning(f"Failed to save plan checkpoint: {e}")

            logging.info(f"Step 2: Filtering plan for example {example_id}")
            example = await FilterAgent.async_filter_plan(model, example, semaphore)

            try:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(example, f)
            except Exception as e:
                logging.error(f"Error saving final checkpoint for {example_id}: {e}")

            if example.get('is_valid', False):
                logging.info(f"Plan for {example_id} PASSED quality check")
            else:
                logging.warning(f"Plan for {example_id} FAILED quality check")
                logging.info(f"Filter analysis: {example.get('filter_analysis', 'No analysis provided')}")

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

    parser = argparse.ArgumentParser(description="Generate and evaluate plans using PlanningAgent.")
    parser.add_argument("--model", type=str, default="Qwen2.5-0.5B-Instruct",
                      help="Specify the model name (default: Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--output_dir", type=str, default="planning_results",
                      help="Specify the output directory (default: planning_results)")
    parser.add_argument("--max_concurrent", type=int, default=5,
                      help="Maximum number of concurrent tasks (default: 5)")
    parser.add_argument("--examples_file", type=str, default="../datasets/Dataset_short.json",
                      help="Path to a JSON file containing examples")
    parser.add_argument("--sample_count", type=int, default=800,
                      help="Number of examples to sample from the dataset (default: 800)")

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
        examples = [
            {
                "prompt": "Sophia is a biologist with a passion for chronicling his vibrant life through weekly diary entries. Captures:\n1) Family member birthday: husband (yourself) (birthday on May 13), my wife (birthday on September 19), my child_1 (birthday on September 06), my father (birthday on July 27), my mother (birthday on June 05), \n2) Attending a fashion week in Paris in week 24-25.\n\n3) Attend a parenting workshop every 5 weeks on weekends, starting from week 13.\n 4) Weekly updates on weather changes, work developments, family life, and other interesting topics.\n5) Use '#*#' to separate each weekly entry (e.g. #*# Week 1 (January 1st - January 7th): \nThe New Year dawned frosty and bright, with temperatures stubbornly clinging to a cool 5°C. The cold didn't deter us from our annual New Year's walk through the nearby park, a tradition that helps us set our intentions for the year ahead. Wrapped in our warmest coats, my wife and I discussed our personal and professional goals for 2018, making plans to nurture both our careers and our family. Back at work this week, the atmosphere in the engineering lab was charged with fresh energy and enthusiasm. The new projects on the horizon promise challenging and innovative solutions that could redefine our approaches. I spent the first few days sketching out the initial phases of a new design that could potentially improve energy efficiency in urban buildings. At home, the post-holiday transition was smooth. We spent the evenings savoring the quiet after the festive buzz, gradually taking down the Christmas decorations. However, we decided to keep some twinkling lights around the living room; their soft glow brings a cozy cheer to the long winter nights. This week also marked the return to school for the kids. They shared stories of their holiday adventures with friends, buzzing with excitement and a bit of holiday hangover. As for the weekend, it brought a gentle snowfall that transformed our neighborhood into a serene winter landscape, perfect for some impromptu snowball fights and hot chocolate by the fireplace. Overall, the week was a blend of reflection, planning, and familial warmth, setting a hopeful tone for the new year)\nGenerate a complete weekly diary for Sophia for the entire year of 2018. Start from January 1st, a Monday, marking the first week, and continue through to December 31st, the end of the 52nd week. Ensure that the diary consists of 52 entries, one for each week. Each diary entry should be at least 200 words. When the design of all 52 weeks is complete, use '*** finished ***' to indicate the end of the document. Ensure clarity and continuity without any interruptions or omissions in the narrative throughout the year. *** started ***\n#*# Week 1 (January 1st - January 7th):",
                "checks_once": {
                    "19": "husband (yourself) birthday",
                    "38": "my wife birthday",
                    "36": "my child_1 birthday",
                    "30": "my father birthday",
                    "23": "my mother birthday"
                },
                "checks_range": {
                    "24": "Attending a fashion week in Paris",
                    "25": "Attending a fashion week in Paris"
                },
                "checks_periodic": {
                    "13": "Attend a parenting workshop",
                    "18": "Attend a parenting workshop",
                    "23": "Attend a parenting workshop",
                    "28": "Attend a parenting workshop",
                    "33": "Attend a parenting workshop",
                    "38": "Attend a parenting workshop",
                    "43": "Attend a parenting workshop",
                    "48": "Attend a parenting workshop"
                },
                "type": "Week",
                "prefix": "#*# Week 1 (January 1st - January 7th):",
                "number": 52
            }
        ]

    for i, example in enumerate(examples):
        example['id'] = f"{example['type']}_{i}"

    tasks = [process_example(args.model, example, semaphore, checkpoint_dir) for example in examples]
    results = await tqdm.gather(*tasks, desc="Generating and evaluating plans")

    has_errors = False
    valid_plans = 0
    invalid_plans = 0

    for result in results:
        if 'error' in result:
            has_errors = True
            logging.error(f"Example {result.get('id', 'unknown')} failed: {result['error']}")
        elif result.get('is_valid', False):
            valid_plans += 1
        else:
            invalid_plans += 1

    final_output_file = os.path.join(output_dir, "planning_results.json")
    logging.info(f"Writing final results to {final_output_file}")
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logging.info("=" * 50)
    logging.info("PLANNING RESULTS SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Total examples processed: {len(results)}")
    logging.info(f"Valid plans: {valid_plans}")
    logging.info(f"Invalid plans: {invalid_plans}")
    logging.info(f"Failed examples: {sum(1 for r in results if 'error' in r)}")
    logging.info("=" * 50)

    logging.info("\nPlan details:")
    for i, result in enumerate(results):
        plan_id = result.get('id', f"Example {i+1}")
        if 'error' in result:
            status = f"ERROR: {result['error']}"
        elif result.get('is_valid', False):
            status = f"VALID: {result.get('filter_result', '')}"
        else:
            status = f"INVALID: {result.get('filter_result', '')}"

        reason = result.get('filter_analysis', 'No analysis')
        if len(reason) > 100:
            reason = reason[:100] + "..."

        logging.info(f"{plan_id}: {status} - {reason}")

    if has_errors:
        logging.warning("Some examples failed to process. Check logs for details.")

    return results

if __name__ == "__main__":
    try:
        results = asyncio.run(main())
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")
        raise
