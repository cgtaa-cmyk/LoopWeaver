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

    max_length: int = field(default=1270,metadata={"help":"Maximum all sequence length."},)
    max_prompt_length: int = field(default=1256,metadata={"help":"Maximum prompt prompt length."},)
    max_passage_length: int = field(default=1024,metadata={"help":"Maximum prompt passage length."},)


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
    one_item = {}

    plan_type = example.get("type", "")

    if plan_type == "Week":
        expert_title = "an expert quality controller"
        plan_name = "weekly plan"
        plan_field = "weekly_plan"
        special_check = "special events are correctly placed in their appropriate weeks"
    elif plan_type == "Floor":
        expert_title = "an expert architect and quality controller"
        plan_name = "floor plan"
        plan_field = "floor_plan"
        special_check = "special facilities are correctly placed on their designated floors"
    elif plan_type == "Menu Week" or plan_type == "Menu":
        expert_title = "an expert chef and quality controller"
        plan_name = "weekly menu plan"
        plan_field = "weekly_plan"
        special_check = "special dishes are correctly planned for their appropriate weeks"
    elif plan_type == "Block":
        expert_title = "an expert urban planner and quality controller"
        plan_name = "city block plan"
        plan_field = "block_plan"
        special_check = "special facilities are correctly placed in their designated blocks"
    else:
        expert_title = "an expert quality controller"
        plan_name = "plan"
        plan_field = "plan"
        special_check = "special requirements are correctly addressed"

    system_prompt = f"""
You are {expert_title}. Your task is to determine if a {plan_name} meets the user requirements.
Given the following user requirements and generated {plan_name}, evaluate if the plan satisfies all requirements.

User requirements:
{{prompt}}

Generated {plan_name}:
{{plan}}

Think step by step:
1. Identify all key requirements specified by the user
2. Check if the {plan_name} addresses each requirement
3. Verify if {special_check}
4. Check for any contradictions or inconsistencies in the plan

Answer only with "YES" if the plan fully satisfies all requirements, or "NO" if it has any deficiencies.
Your final answer (YES/NO):"""

    prompt = example.get("prompt", "")
    plan_content = example.get(plan_field, example.get("chosen", ""))

    if isinstance(plan_content, (dict, list)):
        plan_str = json.dumps(plan_content, indent=2)
    else:
        plan_str = str(plan_content)

    new_prompt = system_prompt.format(prompt=prompt, plan=plan_str)

    messages = [
        {"role": "user", "content": new_prompt}
    ]

    item_input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    one_item["prompt"] = item_input_ids
    one_item["chosen"] = example.get("chosen", "YES")
    one_item["rejected"] = example.get("rejected", "NO")

    return one_item


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
    partial_preprocess = partial(preprocessing,args=data_args,tokenizer=tokenizer)

    def load_and_process_json(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            if isinstance(item.get('chosen'), (list, dict)):
                item['chosen'] = json.dumps(item['chosen'], ensure_ascii=False)
            if isinstance(item.get('rejected'), (list, dict)):
                item['rejected'] = json.dumps(item['rejected'], ensure_ascii=False)

        return data

    from datasets import Dataset

    train_data = load_and_process_json(data_args.train_data_path)
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(partial(preprocessing, args=data_args, tokenizer=tokenizer))

    eval_data = load_and_process_json(data_args.eval_data_path)
    eval_dataset = Dataset.from_list(eval_data)
    eval_dataset = eval_dataset.map(partial(preprocessing, args=data_args, tokenizer=tokenizer))


    dpo_config = DPOConfig(
        output_dir=training_args.output_dir,
        beta=0.1,
        loss_type='sigmoid',
        label_smoothing=0.05,
        use_liger_loss=False,
        sync_ref_model=False,
        ref_model_sync_steps=200,
        ref_model_mixup_alpha=0.5,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        num_train_epochs=training_args.num_train_epochs,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
        learning_rate=5e-6,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        label_names=["chosen", "rejected"],
        ddp_find_unused_parameters=False
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
    )

    dpo_trainer.train()
    dpo_trainer.save_model()
