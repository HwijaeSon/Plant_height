#!/usr/bin/env bash
# Reproduce the final comparison with three seed workers on three GPUs.
# Usage: bash maize/code/run_all.sh [gpu_seed1 gpu_seed2 gpu_seed3]
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/../.." && pwd)
cd "$project_root"
python_bin="$project_root/.venv/bin/python"
result_dir="$project_root/maize/results/chronological_final_seed1_3"
mkdir -p "$result_dir/logs"
gpu_ids=("${1:-1}" "${2:-2}" "${3:-6}")
if [[ "${gpu_ids[0]}" == "${gpu_ids[1]}" || "${gpu_ids[0]}" == "${gpu_ids[2]}" || "${gpu_ids[1]}" == "${gpu_ids[2]}" ]]; then
    echo "Provide three distinct GPU IDs" >&2
    exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2

"$python_bin" -u maize/code/run_experiment.py --models process --device cpu --output "$result_dir" \
    > "$result_dir/logs/process.log" 2>&1

worker_pids=()
for seed in 1 2 3; do
    gpu="${gpu_ids[$((seed-1))]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u maize/code/run_experiment.py \
        --models latent lstm pinn rf --seed "$seed" --output "$result_dir" \
        > "$result_dir/logs/seed${seed}_gpu${gpu}.log" 2>&1 &
    worker_pids+=("$!")
done
failed=0
for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "At least one worker failed; inspect $result_dir/logs" >&2
    exit 1
fi
"$python_bin" maize/code/summarize_results.py --results "$result_dir"
"$python_bin" maize/code/verify_experiment.py
