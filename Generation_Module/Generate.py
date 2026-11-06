import argparse
import logging
import json
import asyncio
import os
from tqdm.asyncio import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LoopWaver_model.Agents.GenerationAgent import GenerationAgent

async def process_example(model, example, semaphore):
    async with semaphore:
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                return await GenerationAgent.async_generate(model, example, semaphore)
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    logging.error(f"Failed to process example {example.get('id', str(example))} after {max_retries} attempts: {e}")
                    raise
                logging.warning(f"Retry {retry_count} for example {example.get('id', str(example))}: {e}")
                await asyncio.sleep(1 * retry_count)

async def main():
    parser = argparse.ArgumentParser(description="Generate text from filtered JSON results")

    parser.add_argument("--model", type=str, help="Specify the model name", required=True)
    parser.add_argument("--input_file", type=str, default="",
                      help="Path to the filtered JSON input file")
    parser.add_argument("--output_dir", type=str, help="Specify the output directory", required=True)

    args = parser.parse_args()
    model = args.model
    input_file = args.input_file

    dataset = []
    logging.info(f"Loading dataset from {input_file}")
    try:
        with open(input_file, 'r', encoding="utf-8") as file:
            dataset = json.load(file)
        logging.info(f"Successfully loaded {len(dataset)} examples from {input_file}")
    except Exception as e:
        logging.error(f"Error loading input file {input_file}: {e}")
        return

    print(f"Processing {len(dataset)} examples")

    input_name = os.path.splitext(os.path.basename(input_file))[0]

    semaphore = asyncio.Semaphore(400)

    tasks = [process_example(model, example, semaphore) for example in dataset]
    final_outputs = await tqdm.gather(*tasks, desc=f"Processing {input_name}")

    for output in final_outputs:
        if isinstance(output, Exception):
            raise output

    output_file = os.path.join(args.output_dir, f"generated_{input_name}.json")
    logging.info(f"Writing output to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(final_outputs, file, indent=2)
    logging.info(f"Processing complete. Results saved to {output_file}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

    asyncio.run(main())
