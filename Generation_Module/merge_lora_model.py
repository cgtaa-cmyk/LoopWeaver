from transformers import AutoModelForCausalLM
from peft import PeftModel
import argparse

def merge_lora_weights(base_model_path, lora_weights_path, output_path):
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(base_model, lora_weights_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print("Saving merged model...")
    model.save_pretrained(output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument("--lora_weights_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")

    args = parser.parse_args()

    base_model_path = args.base_model_path
    lora_weights_path = args.lora_weights_path
    output_path = args.output_path

    merge_lora_weights(base_model_path, lora_weights_path, output_path)
