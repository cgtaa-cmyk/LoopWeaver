import asyncio
import logging
import json
import os
import argparse
from tqdm.asyncio import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LoopWaver_model.Agents.RewardAgent import RolloutRewardAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

async def process_example(example, checkpoint_dir):
    """Process the reward calculation for a single example, including checkpoint and error handling."""

    example_id = example.get('id', f"{example['type']}_{hash(example.get('prompt', ''))}")
    checkpoint_file = os.path.join(checkpoint_dir, f'reward_checkpoint_{example_id}.json')

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                logging.info(f"Loading reward checkpoint for {example_id}")
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading reward checkpoint for {example_id}: {e}")

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            if example.get('is_valid', False) or example.get('filter_response', '').upper() == "YES":
                logging.info(f"Calculating rewards for valid example {example_id}")

                example_with_rewards = RolloutRewardAgent.rollout_rewards(example)

                if 'reward_stats' in example_with_rewards:
                    stats = example_with_rewards['reward_stats']
                    logging.info(f"Reward stats for {example_id}: mean={stats['mean']:.2f}, max={stats['max']:.2f}, min={stats['min']:.2f}")
                    if 'positive_rewards' in stats:
                        logging.info(f"Positive rewards: {stats['positive_rewards']}/{stats.get('total_steps', 0)}")

                try:
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(example_with_rewards, f)
                except Exception as e:
                    logging.error(f"Error saving reward checkpoint for {example_id}: {e}")

                return example_with_rewards
            else:
                logging.info(f"Skipping invalid example {example_id}")
                return example

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                logging.error(f"Failed to process example {example_id} after {max_retries} attempts: {e}")
                example['reward_error'] = str(e)
                return example
            logging.warning(f"Reward calculation attempt {retry_count} failed for example {example_id}: {e}. Retrying...")
            await asyncio.sleep(1 * retry_count)

async def main():
    """Main function: process all examples, supports command-line arguments."""

    parser = argparse.ArgumentParser(description="Calculate rewards for valid plans in planning_results.json")
    parser.add_argument("--input_file", type=str, default="planning_results/planning_results.json",
                      help="Path to planning_results.json file (default: planning_results/planning_results.json)")
    parser.add_argument("--output_dir", type=str, default="reward_results",
                      help="Specify the output directory (default: reward_results)")
    parser.add_argument("--max_concurrent", type=int, default=5,
                      help="Maximum number of concurrent tasks (default: 5)")
    parser.add_argument("--valid_only", action="store_true",
                      help="Process only valid plans")
    parser.add_argument("--type_filter", type=str, default="",
                      help="Filter examples by type (e.g., 'Week', 'Floor', 'Menu Week', 'Block')")

    args = parser.parse_args()

    output_dir = args.output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        logging.info(f"Loaded {len(examples)} examples from {args.input_file}")
    except Exception as e:
        logging.error(f"Error loading input file: {e}")
        return []

    filtered_examples = []
    for example in examples:
        if args.valid_only and not (example.get('is_valid', False) or example.get('filter_response', '').upper() == "YES"):
            continue

        if args.type_filter and example.get('type', '') != args.type_filter:
            continue

        filtered_examples.append(example)

    logging.info(f"Processing {len(filtered_examples)} examples after filtering")

    tasks = [process_example(example, checkpoint_dir) for example in filtered_examples]
    results = await tqdm.gather(*tasks, desc="Calculating rewards")

    processed = 0
    with_rewards = 0
    error_count = 0

    for result in results:
        processed += 1
        if 'reward_error' in result:
            error_count += 1
        elif 'reward_stats' in result:
            with_rewards += 1

    final_output_file = os.path.join(output_dir, "reward_results.json")
    logging.info(f"Writing reward results to {final_output_file}")
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logging.info("=" * 50)
    logging.info("REWARD CALCULATION RESULTS SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Total examples processed: {processed}")
    logging.info(f"Examples with rewards calculated: {with_rewards}")
    logging.info(f"Examples with errors: {error_count}")
    logging.info("=" * 50)

    if with_rewards > 0:
        logging.info("\nReward statistics:")

        mean_rewards = [r['reward_stats']['mean'] for r in results if 'reward_stats' in r]
        max_rewards = [r['reward_stats']['max'] for r in results if 'reward_stats' in r]
        min_rewards = [r['reward_stats']['min'] for r in results if 'reward_stats' in r]

        if mean_rewards:
            avg_mean = sum(mean_rewards) / len(mean_rewards)
            avg_max = sum(max_rewards) / len(max_rewards)
            avg_min = sum(min_rewards) / len(min_rewards)

            logging.info(f"Average mean reward: {avg_mean:.2f}")
            logging.info(f"Average max reward: {avg_max:.2f}")
            logging.info(f"Average min reward: {avg_min:.2f}")

        logging.info("\nPlan details:")
        for result in results:
            plan_id = result.get('id', 'unknown')
            if 'reward_error' in result:
                logging.info(f"{plan_id}: ERROR: {result['reward_error']}")
            elif 'reward_stats' in result:
                stats = result['reward_stats']
                positive_count = stats.get('positive_rewards', 0)
                total_steps = stats.get('total_steps', 0)

                logging.info(f"{plan_id}: Rewards calculated - Mean: {stats['mean']:.2f}, " +
                          f"Max: {stats['max']:.2f}, Min: {stats['min']:.2f}, " +
                          f"Positive: {positive_count}/{total_steps}")

                if 'max_reward_steps' in result and result['max_reward_steps']:
                    max_step = result['max_reward_steps'][0]
                    max_info = ""

                    if 'week_id' in max_step:
                        max_info = f"Week: {max_step['week_id']}"
                    elif 'floor_id' in max_step:
                        max_info = f"Floor: {max_step['floor_id']}"
                    elif 'block_id' in max_step:
                        max_info = f"Block: {max_step['block_id']}"

                    logging.info(f"  Best step: {max_info} (reward: {max_step.get('reward', 0):.2f})")

    return results

if __name__ == "__main__":
    try:
        results = asyncio.run(main())
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")
        raise
