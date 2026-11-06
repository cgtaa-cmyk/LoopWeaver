#!/usr/bin/env bash
#
# LoopWeaver one-click training script: run the planning training (Plan_train) and
# generation training (Gen_train) pipelines in sequence.
#
# The dataset is already split into train/dev (datasets/{train,dev}/data.json); each stage
# runs train and dev separately. DPO training uses the train preference pairs as train_data
# and the dev preference pairs as eval_data.
#
# Usage:
#   bash train_pipeline.sh            # run all (planning + generation)
#   bash train_pipeline.sh planning   # planning training only
#   bash train_pipeline.sh generation # generation training only
#
# Adjust the config section below as needed before running.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================ Config ============================
# Model name (used by Filter/Reward/Generate steps that go through llms.py, e.g. Qwen2.5-7B-Instruct / gpt-4o-mini)
MODEL_NAME="Qwen2.5-7B-Instruct"
# Base model path (used by DPO training / LoRA merge, the real from_pretrained path)
BASE_MODEL_PATH="/path/to/model"

# train / dev dataset
TRAIN_DATA="${ROOT}/datasets/train/data.json"
DEV_DATA="${ROOT}/datasets/dev/data.json"

# Output root directory
OUT="${ROOT}/training_output"

# GPUs used by DPO training (data generation steps go through the vLLM service)
GPU_PLAN=0
GPU_GEN=1
# ================================================================

log() { echo "[$(date '+%F %T')] $*"; }

# Merge LoRA into the base model to produce a directly usable model
merge_lora() {
    local base="$1" lora="$2" out="$3"
    python "$ROOT/Generation_Module/merge_lora_model.py" \
        --base_model_path "$base" \
        --lora_weights_path "$lora" \
        --output_path "$out"
}

# ------------------- Planning training (Plan_train) -------------------
run_planning() {
    local p="$OUT/planning"

    log "=== [Planning] Step1: Filter sub-plans (train / dev) ==="
    python "$ROOT/Planning_Module/Filter_example_use.py" \
        --model "$MODEL_NAME" --examples_file "$TRAIN_DATA" --output_dir "$p/train_output"
    python "$ROOT/Planning_Module/Filter_example_use.py" \
        --model "$MODEL_NAME" --examples_file "$DEV_DATA" --output_dir "$p/dev_output"

    log "=== [Planning] Step2: Compute rewards (train / dev) ==="
    python "$ROOT/Planning_Module/Reward_example_use.py" \
        --input_file "$p/train_output/planning_results.json" --output_dir "$p/reward_train"
    python "$ROOT/Planning_Module/Reward_example_use.py" \
        --input_file "$p/dev_output/planning_results.json" --output_dir "$p/reward_dev"

    log "=== [Planning] Step3: Generate DPO preference pairs (train / dev) ==="
    python "$ROOT/Planning_Module/DPO_example_use.py" \
        --input "$p/reward_train/reward_results.json" --output "$p/train/high_quality_pairs.json"
    python "$ROOT/Planning_Module/DPO_example_use.py" \
        --input "$p/reward_dev/reward_results.json" --output "$p/dev/high_quality_pairs.json"

    log "=== [Planning] Step4: DPO training ==="
    (cd "$ROOT/Planning_Module" && CUDA_VISIBLE_DEVICES="$GPU_PLAN" python DPOTrainer.py \
        --model_name_or_path "$BASE_MODEL_PATH" \
        --train_data_path "$p/train/high_quality_pairs.json" \
        --eval_data_path "$p/dev/high_quality_pairs.json" \
        --output_dir "$p/DPO_output" \
        --logging_dir "$p/DPO_logs" \
        --per_device_train_batch_size 1 \
        --num_train_epochs 10 \
        --save_steps 1000 \
        --eval_steps 200 \
        --logging_steps 10 \
        --use_lora true \
        --bf16 true \
        --optim "adamw_torch")

    log "=== [Planning] Step5: Merge LoRA -> plan_train ==="
    merge_lora "$BASE_MODEL_PATH" "$p/DPO_output" "$p/plan_train_merged"
    log "=== [Planning] done: Plan_train at $p/plan_train_merged ==="
}

# ------------------- Generation training (Gen_train) -------------------
run_generation() {
    local g="$OUT/generation"
    local pt="$OUT/planning/train_output/planning_results.json"
    local pd="$OUT/planning/dev_output/planning_results.json"

    log "=== [Generation] Step1: Filter planning results (train / dev) ==="
    python "$ROOT/Generation_Module/FilterResult.py" \
        --input_file "$pt" --output_file "$g/filter/train.json"
    python "$ROOT/Generation_Module/FilterResult.py" \
        --input_file "$pd" --output_file "$g/filter/dev.json"

    log "=== [Generation] Step2: Generate content from plans (train / dev) ==="
    python "$ROOT/Generation_Module/Generate.py" \
        --model "$MODEL_NAME" --input_file "$g/filter/train.json" --output_dir "$g/generate_train"
    python "$ROOT/Generation_Module/Generate.py" \
        --model "$MODEL_NAME" --input_file "$g/filter/dev.json" --output_dir "$g/generate_dev"

    log "=== [Generation] Step3: Compute generation quality rewards (train / dev) ==="
    python "$ROOT/Generation_Module/Reward_example_use.py" \
        --input_file "$g/generate_train/generated_train.json" --output_dir "$g/reward_train"
    python "$ROOT/Generation_Module/Reward_example_use.py" \
        --input_file "$g/generate_dev/generated_dev.json" --output_dir "$g/reward_dev"

    log "=== [Generation] Step4: Generate DPO preference pairs (train / dev, --format pairs --min_diff 0.5) ==="
    python "$ROOT/Generation_Module/DPO_example_use.py" \
        --input "$g/reward_train/reward_results.json" --output "$g/train/high_quality_pairs.json" \
        --format pairs --min_diff 0.5
    python "$ROOT/Generation_Module/DPO_example_use.py" \
        --input "$g/reward_dev/reward_results.json" --output "$g/dev/high_quality_pairs.json" \
        --format pairs --min_diff 0.5

    log "=== [Generation] Step5: DPO training ==="
    (cd "$ROOT/Generation_Module" && CUDA_VISIBLE_DEVICES="$GPU_GEN" python DPOTrainer.py \
        --model_name_or_path "$BASE_MODEL_PATH" \
        --train_data_path "$g/train/high_quality_pairs.json" \
        --eval_data_path "$g/dev/high_quality_pairs.json" \
        --output_dir "$g/DPO_output" \
        --logging_dir "$g/DPO_logs" \
        --per_device_train_batch_size 1 \
        --num_train_epochs 10 \
        --save_steps 1000 \
        --eval_steps 200 \
        --logging_steps 10 \
        --use_lora true \
        --bf16 true \
        --optim "adamw_torch")

    log "=== [Generation] Step6: Merge LoRA -> gen_train ==="
    merge_lora "$BASE_MODEL_PATH" "$g/DPO_output" "$g/gen_train_merged"
    log "=== [Generation] done: Gen_train at $g/gen_train_merged ==="
}

# ------------------- Main flow -------------------
case "${1:-all}" in
    planning)   run_planning ;;
    generation) run_generation ;;
    all)        run_planning; run_generation ;;
    *) echo "Usage: $0 [planning|generation|all]"; exit 1 ;;
esac

log "All training pipelines finished. Outputs:"
log "  Plan_train: $OUT/planning/plan_train_merged"
log "  Gen_train : $OUT/generation/gen_train_merged"
