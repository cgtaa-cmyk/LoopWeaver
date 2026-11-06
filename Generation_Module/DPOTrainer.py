import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments,PreTrainedModel,AutoConfig)
from functools import partial
import logging
import json
from trl import DPOTrainer, DPOConfig
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional
from datasets import load_dataset, Dataset
import torch
import transformers
import torch.nn as nn
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from peft import PeftConfig, PeftModel
from peft import LoraConfig, TaskType, get_peft_model
import random
logger = logging.getLogger(__name__)
random.seed(42)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    cache_dir: str = field(default="tmp")


@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "Path to the test data."},
    )

    max_length: int = field(default=2048, metadata={"help":"Maximum all sequence length."},)
    max_prompt_length: int = field(default=512, metadata={"help":"Maximum prompt length."},)
    max_response_length: int = field(default=1536, metadata={"help":"Maximum response length."},)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch", metadata={"help": "The optimizer to use."})
    use_lora: bool = field(default=True)
    output_dir : str = field(default=None)
    save_steps : int = field(default=1000)
    eval_steps : int = field(default=200)
    per_device_train_batch_size: int = field(default=1)
    evaluation_strategy: str = field(default='steps')
    logging_steps : int = field(default=10)
    logging_dir : str = field(default=None)
    bf16 : bool = field(default=True)
    num_train_epochs: int = field(default=10)

def load_model_and_tokenizer(
    model_path: str,
    use_lora: bool = True,
    bf16: bool = False,
    fp16: bool = False,
):
    """load model and tokenizer"""

    assert not (bf16 and fp16), "bf16 or fp16, not both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    inference_mode=False,
                )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer

def preprocessing(example, args, tokenizer):
    """Preprocessing function - ensure data types are consistent."""
    one_item = {}

    plan_type = example.get("type", "")

    if plan_type == "Week":
        instruction = "Create a weekly plan based on the following requirements:"
    elif plan_type == "Floor":
        instruction = "Design a floor plan based on the following requirements:"
    elif plan_type == "Menu Week" or plan_type == "Menu":
        instruction = "Create a weekly menu plan based on the following requirements:"
    elif plan_type == "Block":
        instruction = "Design a city block plan based on the following requirements:"
    else:
        instruction = "Create a detailed plan based on the following requirements:"

    prompt = example.get("prompt", "")

    complete_prompt = f"{instruction}\n\n{prompt}\n\nPlease organize your plan with clear sections."

    messages = [
        {"role": "user", "content": complete_prompt}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    one_item["prompt"] = prompt_text
    one_item["chosen"] = example.get("chosen", "")
    one_item["rejected"] = example.get("rejected", "")

    return one_item


def load_and_process_json(json_path):
    """Load and process JSON data, ensuring data types are consistent."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    for item in data:
        processed_item = {}

        processed_item["prompt"] = str(item.get("prompt", ""))

        if "chosen" in item:
            if isinstance(item["chosen"], (dict, list)):
                processed_item["chosen"] = json.dumps(item["chosen"], ensure_ascii=False)
            else:
                processed_item["chosen"] = str(item["chosen"])
        else:
            processed_item["chosen"] = ""

        if "rejected" in item:
            if isinstance(item["rejected"], (dict, list)):
                processed_item["rejected"] = json.dumps(item["rejected"], ensure_ascii=False)
            else:
                processed_item["rejected"] = str(item["rejected"] or "")
        else:
            processed_item["rejected"] = ""

        if "type" in item:
            processed_item["type"] = str(item["type"])

        processed_data.append(processed_item)

    return processed_data


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    logger.info("DATA parameters %s", data_args)

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        use_lora=training_args.use_lora,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
    )

    def load_and_process_json(json_path):
        """Load and process JSON data, ensuring data types are consistent."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []
        for item in data:
            processed_item = {}

            processed_item["prompt"] = str(item.get("prompt", ""))

            if "chosen" in item:
                if isinstance(item["chosen"], (dict, list)):
                    processed_item["chosen"] = json.dumps(item["chosen"], ensure_ascii=False)
                else:
                    processed_item["chosen"] = str(item["chosen"])
            else:
                processed_item["chosen"] = ""

            if "rejected" in item:
                if isinstance(item["rejected"], (dict, list)):
                    processed_item["rejected"] = json.dumps(item["rejected"], ensure_ascii=False)
                else:
                    processed_item["rejected"] = str(item["rejected"] or "")
            else:
                processed_item["rejected"] = ""

            if "type" in item:
                processed_item["type"] = str(item["type"])

            processed_data.append(processed_item)

        return processed_data

    train_data = load_and_process_json(data_args.train_data_path)
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(partial(preprocessing, args=data_args, tokenizer=tokenizer))

    eval_data = load_and_process_json(data_args.eval_data_path)
    eval_dataset = Dataset.from_list(eval_data)
    eval_dataset = eval_dataset.map(partial(preprocessing, args=data_args, tokenizer=tokenizer))

    logger.info("Training data examples:")
    for i in range(min(2, len(train_dataset))):
        logger.info(f"Example {i}:")
        logger.info(f"Prompt: {train_dataset[i]['prompt'][:100]}...")
        logger.info(f"Chosen: {train_dataset[i]['chosen'][:100]}...")
        if train_dataset[i]['rejected']:
            logger.info(f"Rejected: {train_dataset[i]['rejected'][:100]}...")

    dpo_config = DPOConfig(
        output_dir=training_args.output_dir,
        beta=0.1,
        loss_type='sigmoid',
        label_smoothing=0.05,
        use_liger_loss=False,
        sync_ref_model=False,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        num_train_epochs=training_args.num_train_epochs,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
        learning_rate=5e-6,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        max_length=data_args.max_length,
        max_prompt_length=data_args.max_prompt_length,
        max_completion_length=data_args.max_response_length,
        label_names=["chosen", "rejected"],
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"]
    )

    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config if training_args.use_lora else None
    )

    logger.info("Starting training...")
    dpo_trainer.train()

    logger.info(f"Saving model to {training_args.output_dir}")
    dpo_trainer.save_model()
    logger.info("Training complete!")
