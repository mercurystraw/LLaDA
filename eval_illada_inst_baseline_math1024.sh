REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/illada_inst_baseline_math1024/${RUN_ID}}"

GPU_ID=0
IDLE_CHECKS=0

while true; do
    stats=$(nvidia-smi -i "$GPU_ID" \
        --query-gpu=utilization.gpu,memory.used \
        --format=csv,noheader,nounits)

    read -r util mem <<< "$(echo "$stats" | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}')"

    procs=$(nvidia-smi -i "$GPU_ID" \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits |
        awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {n++} END {print n+0}')

    echo "$(date '+%F %T') GPU${GPU_ID}: util=${util}% mem=${mem}MiB processes=${procs}"

    if (( util <= 5 && mem <= 500 && procs == 0 )); then
        IDLE_CHECKS=$((IDLE_CHECKS + 1))
        echo "GPU idle check: ${IDLE_CHECKS}/3"
    else
        IDLE_CHECKS=0
    fi

    if (( IDLE_CHECKS >= 3 )); then
        break
    fi

    sleep 30
done

CUDA_VISIBLE_DEVICES="$GPU_ID" \
MATH_MAX_GEN=1024 \
OUTPUT_ROOT="$OUTPUT_ROOT" \
bash "${REPO_ROOT}/eval_illada_inst_baseline.sh" math500
