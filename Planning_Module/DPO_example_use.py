import asyncio
import logging
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LoopWaver_model.Agents.DPODataAgent import DPODataAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

async def main():
    """Main function: process command-line arguments and generate DPO data."""

    parser = argparse.ArgumentParser(description="Generate DPO training data from reward results")
    parser.add_argument("--input", type=str, default="reward_results/reward_results.json",
                      help="Path to reward results JSON file")
    parser.add_argument("--output", type=str, default="dpo_data/dpo_pairs.json",
                      help="Output path for DPO training data")
    parser.add_argument("--format", type=str, choices=['pairs', 'training'], default='training',
                      help="Output format: 'pairs' for detailed info, 'training' for training format")
    parser.add_argument("--min_diff", type=float, default=0.2,
                      help="Minimum reward difference between chosen and rejected examples")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = await DPODataAgent.async_generate_dpo_data(
        args.input, args.output, args.format, args.min_diff
    )

    logging.info("=" * 50)
    logging.info("DPO DATA GENERATION SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Input file: {args.input}")
    logging.info(f"Output file: {args.output}")
    logging.info(f"Format: {args.format}")
    logging.info(f"Generated pairs: {len(results)}")

    if results:
        if args.format == 'pairs':
            reward_diffs = [item['chosen_reward'] - item['rejected_reward'] for item in results]
            avg_diff = sum(reward_diffs) / len(reward_diffs)
            max_diff = max(reward_diffs)
            min_diff = min(reward_diffs)

            logging.info(f"Average reward difference: {avg_diff:.2f}")
            logging.info(f"Max reward difference: {max_diff:.2f}")
            logging.info(f"Min reward difference: {min_diff:.2f}")

        type_counts = {}
        for item in results:
            item_type = item.get('type', 'unknown')
            type_counts[item_type] = type_counts.get(item_type, 0) + 1

        logging.info("\nData distribution by type:")
        for type_name, count in type_counts.items():
            logging.info(f"  {type_name}: {count} pairs ({count/len(results)*100:.1f}%)")

    logging.info("=" * 50)
    return results

if __name__ == "__main__":
    try:
        results = asyncio.run(main())
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")
        raise
