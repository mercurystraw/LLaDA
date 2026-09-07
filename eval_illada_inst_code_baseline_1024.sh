#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash eval_illada_inst_code_baseline_1024.sh all 0
#   bash eval_illada_inst_code_baseline_1024.sh humaneval 1
#   bash eval_illada_inst_code_baseline_1024.sh mbpp 2
#   LIMIT=10 bash eval_illada_inst_code_baseline_1024.sh humaneval 0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

TASK="${1:-all}"
GPU_INDEX="${2:-${GPU_INDEX:-0}}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/model_weights/iLLaDA-8B-Instruct}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29505}"
LIMIT="${LIMIT:-full}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/illada_inst_code_baseline_1024/${RUN_ID}}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

case "${TASK}" in
    all|humaneval|mbpp) ;;
    *)
        echo "Usage: bash $0 [all|humaneval|mbpp] [GPU_INDEX]" >&2
        exit 2
        ;;
esac

if [[ ! "${GPU_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "GPU_INDEX must be a non-negative integer, got: ${GPU_INDEX}" >&2
    exit 2
fi

if [[ "${LIMIT}" != "full" && ! "${LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LIMIT must be a positive integer or 'full', got: ${LIMIT}" >&2
    exit 2
fi

if [[ ! -f "${MODEL_PATH}/config.json" || ! -f "${MODEL_PATH}/model.safetensors.index.json" ]]; then
    echo "iLLaDA-Instruct weights not found at: ${MODEL_PATH}" >&2
    exit 2
fi

if ! command -v accelerate >/dev/null 2>&1; then
    echo "accelerate is not available. Activate the illada environment first." >&2
    exit 2
fi

limit_args=()
if [[ "${LIMIT}" != "full" ]]; then
    limit_args=(--limit "${LIMIT}")
fi

launch_args=(
    --num_processes 1
    --main_process_port "${MAIN_PROCESS_PORT}"
)

echo "iLLaDA-Instruct code-generation baseline (max_gen=1024)"
echo "  Task:   ${TASK}"
echo "  Model:  ${MODEL_PATH}"
echo "  GPU:    ${GPU_INDEX}"
echo "  Limit:  ${LIMIT}"
echo "  Output: ${OUTPUT_ROOT}"
echo "  Warning: generated Python code will be executed for pass@1 evaluation."

if [[ "${TASK}" == "all" || "${TASK}" == "humaneval" ]]; then
    echo "Running HumanEval reasoning evaluation (max_gen=1024, steps=32, block=32)..."
    PYTHONPATH="${REPO_ROOT}/eval_instruct${PYTHONPATH:+:${PYTHONPATH}}" \
        accelerate launch "${launch_args[@]}" eval_illada.py \
        --model illada_dist \
        --model_args "model_path=${MODEL_PATH},steps=32,block_length=32,var=True,end_think_logit_boost=100,end_think_boost_power=1,gen_length=1024,throughput_output=${OUTPUT_ROOT}/humaneval/throughput.json" \
        --tasks humaneval_reasoning \
        --device cuda \
        --batch_size 1 \
        --num_fewshot 0 \
        --apply_chat_template \
        --confirm_run_unsafe_code \
        --output_path "${OUTPUT_ROOT}/humaneval" \
        --log_samples \
        "${limit_args[@]}"
fi

if [[ "${TASK}" == "all" || "${TASK}" == "mbpp" ]]; then
    echo "Running MBPP reasoning evaluation (max_gen=1024, steps=16, block=16)..."
    PYTHONPATH="${REPO_ROOT}/eval_instruct${PYTHONPATH:+:${PYTHONPATH}}" \
        accelerate launch "${launch_args[@]}" eval_illada.py \
        --model illada_dist \
        --model_args "model_path=${MODEL_PATH},steps=16,block_length=16,var=True,end_think_logit_boost=100,end_think_boost_power=3,gen_length=1024,throughput_output=${OUTPUT_ROOT}/mbpp/throughput.json" \
        --tasks mbpp_reasoning \
        --device cuda \
        --batch_size 1 \
        --num_fewshot 0 \
        --apply_chat_template \
        --confirm_run_unsafe_code \
        --output_path "${OUTPUT_ROOT}/mbpp" \
        --log_samples \
        "${limit_args[@]}"
fi

echo "Code evaluation finished. Results: ${OUTPUT_ROOT}"
