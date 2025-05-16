#!/bin/bash



total_workers=128

for i in $(seq 0 $((total_workers - 1))); do
    params="--worker_id $i --total_workers $total_workers"
    echo "Running with params: $params"
    sbatch \
        --job-name=data-${i} \
        --output=dataset_logs/lean-trees-${i}-%j.out \
        --error=dataset_logs/lean-trees-${i}-%j.err \
        --partition=cpu-troja \
        --cpus-per-task=2 \
        run ~/troja/project_1/venv-amd/bin/python3.12 src/lean_trees/dataset/tree_dataset.py generate $params
done
