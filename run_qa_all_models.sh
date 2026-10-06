#!/usr/bin/env bash

# Run the same QA experiment for ViT5, BARTpho, mT5 and mBART.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODELS=(
    "vit5|VietAI/vit5-base"
    "bartpho|vinai/bartpho-syllable-base"
    "mt5|google/mt5-base"
    "mbart|facebook/mbart-large-50-many-to-many-mmt"
)

for spec in "${MODELS[@]}"; do
    MODEL_TAG="${spec%%|*}"
    MODEL_NAME="${spec#*|}"
    echo
    echo "============================================================"
    echo "Running $MODEL_TAG ($MODEL_NAME)"
    echo "============================================================"
    MODEL_NAME="$MODEL_NAME" \
    OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/qa_models}/$MODEL_TAG" \
    ./run_qa.sh
done

echo
echo "All four model experiments completed."
