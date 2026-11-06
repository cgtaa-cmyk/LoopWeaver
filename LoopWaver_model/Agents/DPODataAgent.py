import json
import random
import logging
import os
from tqdm import tqdm

class DPODataAgent:
    """Agent that generates DPO training data pairs from reward scoring results.
    High-reward steps are used as positive samples and low-reward steps as negative samples to form preference pairs."""

    @staticmethod
    def read_json(file_path):
        """Read a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_jsonl(data, file_path):
        """Save data as JSONL."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    @staticmethod
    def save_json(data, file_path):
        """Save data as JSON."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def extract_content_text(step, plan_type):
        """Extract content text according to the plan type."""
        if plan_type == 'Week' or plan_type.startswith('Menu'):
            if 'events' in step:
                if isinstance(step['events'], list):
                    if all(isinstance(e, dict) for e in step['events']):
                        return "\n".join([f"{e.get('event', '')}: {e.get('details', '')}" for e in step['events']])
                    else:
                        return "\n".join(step['events'])
                else:
                    return str(step['events'])
            elif 'dishes' in step:
                if isinstance(step['dishes'], list):
                    return "\n".join(str(dish) for dish in step['dishes'])
                else:
                    return str(step['dishes'])
            else:
                return str(step)

        elif plan_type == 'Floor':
            if 'purpose' in step:
                return f"{step.get('floor_id', '')}: {step['purpose']}"
            else:
                return str(step)

        elif plan_type == 'Block':
            if 'use' in step:
                return f"{step.get('block_id', '')}: {step['use']}"
            else:
                return str(step)

        return str(step)

    @staticmethod
    def extract_dpo_pairs(reward_results, min_reward_diff=0.2, random_seed=42):
        """Extract positive/negative sample pairs from reward scores."""
        random.seed(random_seed)
        dpo_pairs = []
        valid_count = 0

        for item in tqdm(reward_results, desc="Extracting DPO pairs"):
            if not item.get('is_valid') and not item.get('filter_response', '').upper() == 'YES':
                continue
            if 'reward_stats' not in item:
                continue

            valid_count += 1

            plan_type = item.get('type', '')

            plan_key = None
            for key in ["weekly_plan", "floor_plan", "block_plan",
                       "filtered_weekly_plan", "filtered_floor_plan", "filtered_block_plan"]:
                if key in item and isinstance(item[key], list) and item[key]:
                    plan_key = key
                    break

            if not plan_key:
                continue

            max_steps = item.get('max_reward_steps', [])
            min_steps = item.get('min_reward_steps', [])

            if not max_steps or not min_steps or max_steps[0].get('reward', 0) == min_steps[0].get('reward', 0):
                continue

            reward_diff = max_steps[0].get('reward', 0) - min_steps[0].get('reward', 0)
            if reward_diff < min_reward_diff:
                logging.debug(f"Skipping pair with small reward diff: {reward_diff}")
                continue

            chosen_step = random.choice(max_steps)
            rejected_step = random.choice(min_steps)

            dpo_sample = {
                'id': item.get('id', f"{plan_type}_{valid_count}"),
                'prompt': item.get('prompt', ''),
                'type': plan_type,
                'chosen': chosen_step,
                'rejected': rejected_step,
                'chosen_reward': float(chosen_step.get('reward', 0)),
                'rejected_reward': float(rejected_step.get('reward', 0)),
                'chosen_advantage': float(chosen_step.get('advantage', 0)),
                'rejected_advantage': float(rejected_step.get('advantage', 0))
            }

            for check_key in ['checks_once', 'checks_range', 'checks_periodic']:
                if check_key in item:
                    dpo_sample[check_key] = item[check_key]

            dpo_pairs.append(dpo_sample)

        logging.info(f"Generated {len(dpo_pairs)} valid DPO pairs from {valid_count} valid plans")
        return dpo_pairs

    @staticmethod
    def format_for_training(dpo_pairs):
        """Format DPO pairs into the training format."""
        training_data = []

        for pair in dpo_pairs:
            chosen_text = DPODataAgent.extract_content_text(pair['chosen'], pair['type'])
            rejected_text = DPODataAgent.extract_content_text(pair['rejected'], pair['type'])

            sample = {
                'id': pair['id'],
                'prompt': pair['prompt'],
                'chosen': chosen_text,
                'rejected': rejected_text,
                'type': pair['type']
            }
            training_data.append(sample)

        return training_data

    @staticmethod
    def generate_dpo_data(reward_file, output_file, format_type='training', min_reward_diff=0.2):
        """
        Main entry function to generate DPO training data.

        Args:
        - reward_file: path to the reward results file
        - output_file: output file path
        - format_type: output format, 'pairs' for detailed info, 'training' for the training format
        - min_reward_diff: minimum reward difference, pairs below this are filtered

        Returns:
        - the list of generated DPO data
        """
        try:
            reward_results = DPODataAgent.read_json(reward_file)
            logging.info(f"Loaded {len(reward_results)} examples from {reward_file}")
        except Exception as e:
            logging.error(f"Error reading reward file: {e}")
            return []

        dpo_pairs = DPODataAgent.extract_dpo_pairs(reward_results, min_reward_diff)

        if format_type == 'training':
            training_data = DPODataAgent.format_for_training(dpo_pairs)
            output_file = output_file.replace('.jsonl', '.json')
            DPODataAgent.save_json(training_data, output_file)
            logging.info(f"Saved {len(training_data)} training examples to {output_file}")
            return training_data
        else:
            output_file = output_file.replace('.jsonl', '.json')
            DPODataAgent.save_json(dpo_pairs, output_file)
            logging.info(f"Saved {len(dpo_pairs)} detailed DPO pairs to {output_file}")
            return dpo_pairs

    @staticmethod
    async def async_generate_dpo_data(reward_file, output_file, format_type='training', min_reward_diff=0.2):
        """Asynchronous version of the DPO data generation function."""
        return DPODataAgent.generate_dpo_data(reward_file, output_file, format_type, min_reward_diff)
