#!/usr/bin/env bash

# Run the four paper-inspired extractive QA experiments with one command.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
METHOD_LIST="${METHODS:-cog dfgn hgn qanet}"
SEEDS="${SEEDS:-42}"
ENCODER="${ENCODER:-xlm-roberta-base}"
TRACK="${TRACK:-true_multihop}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/runs/extractive}"
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:-0}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
PASSAGE_LOSS_WEIGHT="${PASSAGE_LOSS_WEIGHT:-0.25}"
MAX_PASSAGES="${MAX_PASSAGES:-6}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SKIP_BERTSCORE="${SKIP_BERTSCORE:-1}"
EVALUATE_TEST="${EVALUATE_TEST:-0}"
read -r -a METHOD_ARRAY <<< "$METHOD_LIST"
read -r -a SEED_ARRAY <<< "$SEEDS"
if [[ ${#METHOD_ARRAY[@]} -eq 0 || ${#SEED_ARRAY[@]} -eq 0 ]]; then
    echo "METHODS and SEEDS must each contain at least one value." >&2
    exit 2
fi

if [[ ! -f "$ROOT_DIR/dataset/QA/extractive/$TRACK/train.jsonl" || \
      ! -f "$ROOT_DIR/dataset/QA/extractive/$TRACK/dev.jsonl" || \
      ! -f "$ROOT_DIR/dataset/QA/extractive/manifest.json" ]]; then
    echo "[1/2] Preparing weak extractive labels..."
    "$PYTHON" -m qa.prepare_extractive
else
    echo "[1/2] Extractive data already exists; skipping preparation."
fi

for SEED in "${SEED_ARRAY[@]}"; do
    for METHOD in "${METHOD_ARRAY[@]}"; do
        RUN_DIR="$OUTPUT_ROOT/${TRACK}_${METHOD}_seed${SEED}"
        echo
        echo "============================================================"
        echo "method=$METHOD encoder=$ENCODER track=$TRACK seed=$SEED"
        echo "epochs=$EPOCHS max_steps=$MAX_STEPS batch=$BATCH_SIZE accumulation=$GRADIENT_ACCUMULATION"
        echo "============================================================"

        TRAIN_ARGS=(
            -m qa.train_extractive
            --method "$METHOD"
            --encoder "$ENCODER"
            --track "$TRACK"
            --output-dir "$OUTPUT_ROOT"
            --epochs "$EPOCHS"
            --max-steps "$MAX_STEPS"
            --max-train-examples "$MAX_TRAIN_EXAMPLES"
            --batch-size "$BATCH_SIZE"
            --gradient-accumulation "$GRADIENT_ACCUMULATION"
            --learning-rate "$LEARNING_RATE"
            --passage-loss-weight "$PASSAGE_LOSS_WEIGHT"
            --max-passages "$MAX_PASSAGES"
            --max-length "$MAX_LENGTH"
            --save-steps "$SAVE_STEPS"
            --seed "$SEED"
        )
        if [[ -f "$RUN_DIR/COMPLETE" ]]; then
            echo "Completed run found; retaining checkpoint and skipping training."
        else
            "$PYTHON" "${TRAIN_ARGS[@]}"
        fi

        EVAL_ARGS=(
            -m qa.evaluate_extractive
            --checkpoint "$RUN_DIR"
            --method "$METHOD"
            --data "$ROOT_DIR/dataset/QA/extractive/$TRACK/dev.jsonl"
            --output "$RUN_DIR/dev_metrics.json"
        )
        if [[ "$SKIP_BERTSCORE" == "1" ]]; then
            EVAL_ARGS+=(--skip-bertscore)
        fi
        "$PYTHON" "${EVAL_ARGS[@]}"

        if [[ "$EVALUATE_TEST" == "1" ]]; then
            TEST_ARGS=(
                -m qa.evaluate_extractive
                --checkpoint "$RUN_DIR"
                --method "$METHOD"
                --data "$ROOT_DIR/dataset/QA/extractive/$TRACK/test.jsonl"
                --output "$RUN_DIR/test_metrics.json"
            )
            if [[ "$SKIP_BERTSCORE" == "1" ]]; then
                TEST_ARGS+=(--skip-bertscore)
            fi
            "$PYTHON" "${TEST_ARGS[@]}"
        fi
    done
done

SUMMARY_ARGS=(
    -m qa.summarize_extractive
    --output-dir "$OUTPUT_ROOT"
    --track "$TRACK"
    --methods "${METHOD_ARRAY[@]}"
    --seeds "${SEED_ARRAY[@]}"
    --splits dev
    --expected-encoder "$ENCODER"
    --expected-epochs "$EPOCHS"
    --expected-max-steps "$MAX_STEPS"
    --expected-max-train-examples "$MAX_TRAIN_EXAMPLES"
    --expected-batch-size "$BATCH_SIZE"
    --expected-gradient-accumulation "$GRADIENT_ACCUMULATION"
    --expected-learning-rate "$LEARNING_RATE"
    --expected-passage-loss-weight "$PASSAGE_LOSS_WEIGHT"
    --expected-max-passages "$MAX_PASSAGES"
    --expected-max-length "$MAX_LENGTH"
    --expected-save-steps "$SAVE_STEPS"
)
if [[ "$EVALUATE_TEST" == "1" ]]; then
    SUMMARY_ARGS+=(test)
fi
"$PYTHON" "${SUMMARY_ARGS[@]}"

echo
echo "All extractive experiments completed with shared settings and paired seeds."
