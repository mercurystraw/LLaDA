#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash eval_illada_inst_baseline.sh             # full GSM8K + MATH-500 evaluation
#   bash eval_illada_inst_baseline.sh gsm8k
#   bash eval_illada_inst_baseline.sh math500
#   LIMIT=10 bash eval_illada_inst_baseline.sh    # smoke test
#   GSM8K_MAX_GEN=2048 MATH_MAX_GEN=4096 bash eval_illada_inst_baseline.sh
#   NUM_PROCESSES=8 bash eval_illada_inst_baseline.sh gsm8k  # optional multi-GPU

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

TASK="${1:-all}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/model_weights/iLLaDA-8B-Instruct}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29503}"
LIMIT="${LIMIT:-full}"
GSM8K_MAX_GEN="${GSM8K_MAX_GEN:-2048}"
MATH_MAX_GEN="${MATH_MAX_GEN:-4096}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/illada_inst_baseline/${RUN_ID}}"

export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "${TASK}" in
    all|gsm8k|math500) ;;
    math) TASK="math500" ;;
    *)
        echo "Usage: bash $0 [all|gsm8k|math500]" >&2
        exit 2
        ;;
esac

if [[ ! "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_PROCESSES must be a positive integer, got: ${NUM_PROCESSES}" >&2
    exit 2
fi

if [[ "${LIMIT}" != "full" && ! "${LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LIMIT must be a positive integer or 'full', got: ${LIMIT}" >&2
    exit 2
fi

for max_gen in "${GSM8K_MAX_GEN}" "${MATH_MAX_GEN}"; do
    if [[ ! "${max_gen}" =~ ^[1-9][0-9]*$ ]] || (( max_gen % 32 != 0 )); then
        echo "Generation limits must be positive multiples of 32, got: ${max_gen}" >&2
        exit 2
    fi
done

if [[ ! -f "${MODEL_PATH}/config.json" || ! -f "${MODEL_PATH}/model.safetensors.index.json" ]]; then
    echo "iLLaDA-Instruct weights not found at: ${MODEL_PATH}" >&2
    exit 2
fi

if ! command -v accelerate >/dev/null 2>&1; then
    echo "accelerate is not available. Activate the illada environment first." >&2
    exit 2
fi

launch_args=(
    --num_processes "${NUM_PROCESSES}"
    --main_process_port "${MAIN_PROCESS_PORT}"
)
if (( NUM_PROCESSES > 1 )); then
    launch_args=(--multi_gpu "${launch_args[@]}")
fi

limit_args=()
if [[ "${LIMIT}" != "full" ]]; then
    limit_args=(--limit "${LIMIT}")
fi

common_model_args="model_path=${MODEL_PATH},steps=32,block_length=32,var=True,end_think_logit_boost=100,end_think_boost_power=3"

echo "iLLaDA-Instruct zero-shot baseline"
echo "  Task:      ${TASK}"
echo "  Model:     ${MODEL_PATH}"
echo "  GPU:       ${CUDA_VISIBLE_DEVICES}"
echo "  Processes: ${NUM_PROCESSES}"
echo "  GSM8K max generation: ${GSM8K_MAX_GEN}"
echo "  MATH max generation:  ${MATH_MAX_GEN}"
echo "  Limit:     ${LIMIT}"
echo "  Output:    ${OUTPUT_ROOT}"

if [[ "${TASK}" == "all" || "${TASK}" == "gsm8k" ]]; then
    echo "Running GSM8K reasoning evaluation (max_gen=${GSM8K_MAX_GEN}, EOS early-stop)..."
    PYTHONPATH="${REPO_ROOT}/eval_instruct${PYTHONPATH:+:${PYTHONPATH}}" \
        accelerate launch "${launch_args[@]}" eval_illada.py \
        --model illada_dist \
        --model_args "${common_model_args},gen_length=${GSM8K_MAX_GEN},throughput_output=${OUTPUT_ROOT}/gsm8k/max_gen_${GSM8K_MAX_GEN}_stop_eos/throughput.json" \
        --tasks gsm8k_cot_reasoning \
        --device cuda \
        --batch_size 1 \
        --num_fewshot 0 \
        --apply_chat_template \
        --output_path "${OUTPUT_ROOT}/gsm8k/max_gen_${GSM8K_MAX_GEN}_stop_eos" \
        --log_samples \
        "${limit_args[@]}"
fi

if [[ "${TASK}" == "all" || "${TASK}" == "math500" ]]; then
    echo "Running MATH-500 evaluation (max_gen=${MATH_MAX_GEN}, EOS early-stop)..."
    env -u PYTHONPATH accelerate launch "${launch_args[@]}" eval_illada.py \
        --model illada_dist \
        --model_args "${common_model_args},gen_length=${MATH_MAX_GEN},throughput_output=${OUTPUT_ROOT}/math500/max_gen_${MATH_MAX_GEN}_stop_eos/throughput.json" \
        --tasks minerva_math500 \
        --device cuda \
        --batch_size 1 \
        --num_fewshot 0 \
        --apply_chat_template \
        --output_path "${OUTPUT_ROOT}/math500/max_gen_${MATH_MAX_GEN}_stop_eos" \
        --log_samples \
        "${limit_args[@]}"
fi

echo "Evaluation finished. Results: ${OUTPUT_ROOT}"
