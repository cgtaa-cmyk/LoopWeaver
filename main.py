import argparse
import logging
from LoopWaver_model.LoopWeaver import LoopWeaver
from LoopWaver_model.BaselineGen import BaselineGen
import json
import asyncio
import hashlib
import os
from tqdm.asyncio import tqdm

def _stable_example_key(example):
    identity = {
        "id": example.get("id"),
        "type": example.get("type"),
        "number": example.get("number"),
        "prompt": example.get("prompt"),
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


async def process_example(model, example, semaphore, checkpoint_dir, generator_type="loopweaver",
                          plan_model=None, gen_model=None):
    example_id = example.get('id', str(example))
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_{_stable_example_key(example)}.json')
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading checkpoint for {example_id}: {e}")

    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if generator_type == "loopweaver":
                processed_example = await LoopWeaver.async_generate(plan_model, gen_model, example, semaphore)
            else:
                processed_example = await BaselineGen.async_generate(model, example, semaphore)
            
            try:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_example, f)
            except Exception as e:
                logging.error(f"Error saving checkpoint for {example_id}: {e}")
            
            return processed_example
            
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                logging.error(f"Failed to process example {example_id} after {max_retries} attempts: {e}")
                raise
            logging.warning(f"Attempt {retry_count} failed for example {example_id}: {e}. Retrying...")
            await asyncio.sleep(1 * retry_count) 

async def main():
    parser = argparse.ArgumentParser(description="Example script with command-line arguments.")
    
    parser.add_argument("--generator", type=str, choices=["loopweaver", "base"], default="loopweaver",
                      help="'loopweaver' = Plan_train -> Gen_train; 'base' = direct generation (default: loopweaver)")
    parser.add_argument("--model", type=str, default=None,
                      help="base model name (required when generator=base)")
    parser.add_argument("--plan_model", type=str, default=None,
                      help="LoopWeaver planning model name Plan_train (required when generator=loopweaver)")
    parser.add_argument("--gen_model", type=str, default=None,
                      help="LoopWeaver generation model name Gen_train (required when generator=loopweaver)")
    parser.add_argument("--dataset_dir", type=str, help="Specify the dataset directory", required=True)
    parser.add_argument("--output_dir", type=str, help="Specify the output directory", required=True)

    args = parser.parse_args()
    generator_type = args.generator
    model = args.model
    plan_model = args.plan_model
    gen_model = args.gen_model
    dataset_dir = args.dataset_dir

    if generator_type == "loopweaver":
        if not plan_model or not gen_model:
            parser.error("--generator loopweaver requires --plan_model and --gen_model")
    else:
        if not model:
            parser.error("--generator base requires --model")
    
    dataset = []
    logging.info(f"Loading dataset from {dataset_dir}")
    with open(dataset_dir, 'r', encoding="utf-8") as file:
        dataset = json.load(file)

    print(len(dataset))
    
    dataset_name = os.path.splitext(os.path.basename(dataset_dir))[0]
    if generator_type == "loopweaver":
        model_name = f"{os.path.basename(plan_model)}_{os.path.basename(gen_model)}"
    else:
        model_name = os.path.basename(model)
    checkpoint_dir = os.path.join("longGenBench_output", model_name, f"{generator_type}_checkpoints_{dataset_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(100)

    tasks = [process_example(model, example, semaphore, checkpoint_dir, generator_type,
                             plan_model, gen_model) for example in dataset]
    final_outputs = await tqdm.gather(*tasks, desc=f"Processing {dataset_name}")
    
    for output in final_outputs:
        if isinstance(output, Exception):
            raise output
    
    logging.info(f"Writing output to {args.output_dir}")
    with open(args.output_dir, 'w') as file:
        json.dump(final_outputs, file)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    asyncio.run(main())