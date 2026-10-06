#!/usr/bin/env bash

# One-command benchmark for sparse, dense, hybrid and paper-inspired IR methods.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
"$PYTHON" -c 'import numpy, sqlite3, torch, transformers; sqlite3.connect(":memory:").execute("create virtual table f using fts5(t)")' 2>/dev/null || {
    echo "IR dependencies or SQLite FTS5 are missing. Install requirements-qa.txt and use SQLite built with FTS5." >&2
    exit 2
}
METHOD_LIST="${METHODS:-bm25 dense hybrid mdr m3 baleen mopo gmr ircot}"
SEEDS="${SEEDS:-42}"
TRACK="${TRACK:-true_multihop}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/runs/ir}"
CORPUS="${CORPUS:-$ROOT_DIR/dataset/IR/experiments/corpus.jsonl}"
ENCODER="${ENCODER:-xlm-roberta-base}"
GMR_MODEL="${GMR_MODEL:-google/mt5-small}"
IRCOT_REASONER="${IRCOT_REASONER:-google/flan-t5-small}"
EPOCHS="${EPOCHS:-2}"
GMR_EPOCHS="${GMR_EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:-0}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
SAVE_STEPS="${SAVE_STEPS:-100}"
TOP_K="${TOP_K:-100}"
SEARCH_DEPTH="${SEARCH_DEPTH:-100}"
MAX_HOPS="${MAX_HOPS:-3}"
EVALUATE_TEST="${EVALUATE_TEST:-0}"
EVALUATE_QA="${EVALUATE_QA:-0}"
QA_CHECKPOINT="${QA_CHECKPOINT:-}"
QA_METHOD="${QA_METHOD:-pathfid}"
QA_MODEL_NAME="${QA_MODEL_NAME:-}"
QA_TOP_K="${QA_TOP_K:-10}"
QA_MAX_SOURCE_LENGTH="${QA_MAX_SOURCE_LENGTH:-2048}"
QA_MAX_TARGET_LENGTH="${QA_MAX_TARGET_LENGTH:-1024}"
QA_MAX_PASSAGE_TOKENS="${QA_MAX_PASSAGE_TOKENS:-256}"
SKIP_BERTSCORE="${SKIP_BERTSCORE:-1}"

read -r -a METHOD_ARRAY <<< "$METHOD_LIST"
read -r -a SEED_ARRAY <<< "$SEEDS"
if [[ ${#METHOD_ARRAY[@]} -eq 0 || ${#SEED_ARRAY[@]} -eq 0 ]]; then
    echo "METHODS and SEEDS must each contain at least one value." >&2
    exit 2
fi
if [[ "$EVALUATE_QA" == "1" && ! -d "$QA_CHECKPOINT" ]]; then
    echo "EVALUATE_QA=1 requires QA_CHECKPOINT to name one fixed, trained QA reader." >&2
    exit 2
fi
NEEDS_NEURAL=0
for METHOD in "${METHOD_ARRAY[@]}"; do
    case "$METHOD" in
        dense|hybrid|mdr|m3|baleen|mopo|gmr|ircot) NEEDS_NEURAL=1 ;;
    esac
done
if [[ "$NEEDS_NEURAL" == "1" && "$($PYTHON -c 'import torch; print(int(torch.cuda.is_available()))')" == "0" ]]; then
    echo "Warning: CUDA is unavailable; neural IR methods will run on CPU and may be slow." >&2
fi

if [[ ! -s "$CORPUS" || ! -f "$(dirname -- "$CORPUS")/manifest.json" || \
      ! -s "$(dirname -- "$CORPUS")/$TRACK/dev_queries.jsonl" || \
      ! -s "$(dirname -- "$CORPUS")/$TRACK/dev_qrels.tsv" ]]; then
    echo "[1/3] Preparing shared article-level corpus and query/qrels tracks..."
    "$PYTHON" -m ir.prepare --output-dir "$(dirname -- "$CORPUS")"
else
    echo "[1/3] Shared corpus exists; skipping preparation."
fi

train_dense_method() {
    local method="$1"
    local seed="$2"
    local model_dir="$OUTPUT_ROOT/models/${TRACK}_${method}_seed${seed}"
    if [[ -f "$model_dir/COMPLETE" ]]; then
        echo "Completed $method model found; retaining its checkpoint."
        return
    fi
    "$PYTHON" -m ir.train_dense \
        --method "$method" --track "$TRACK" --data-dir "$ROOT_DIR/dataset/IR/experiments" \
        --corpus "$CORPUS" --output-root "$OUTPUT_ROOT" --encoder "$ENCODER" \
        --epochs "$EPOCHS" --max-steps "$MAX_STEPS" \
        --max-train-examples "$MAX_TRAIN_EXAMPLES" --batch-size "$BATCH_SIZE" \
        --gradient-accumulation "$GRADIENT_ACCUMULATION" --learning-rate "$LEARNING_RATE" \
        --max-length "$MAX_LENGTH" --save-steps "$SAVE_STEPS" --seed "$seed"
}

run_retrieval() {
    local method="$1"
    local split="$2"
    local seed="$3"
    "$PYTHON" -m ir.retrieve \
        --method "$method" --split "$split" --track "$TRACK" \
        --data-dir "$ROOT_DIR/dataset/IR/experiments" --corpus "$CORPUS" \
        --output-root "$OUTPUT_ROOT" --encoder "$ENCODER" \
        --top-k "$TOP_K" --search-depth "$SEARCH_DEPTH" --max-hops "$MAX_HOPS" \
        --max-length "$MAX_LENGTH" --seed "$seed" --reasoner-model "$IRCOT_REASONER"
    if [[ "$EVALUATE_QA" == "1" ]]; then
        local rankings_path="$OUTPUT_ROOT/runs/${TRACK}_${method}_seed${seed}/${split}_rankings.jsonl"
        local qa_output="$OUTPUT_ROOT/runs/${TRACK}_${method}_seed${seed}/${split}_qa_metrics_top${QA_TOP_K}.json"
        QA_ARGS=(
            -m ir.evaluate_qa --rankings "$rankings_path" --retrieval-method "$method"
            --seed "$seed" --split "$split" --track "$TRACK"
            --data-dir "$ROOT_DIR/dataset/IR/experiments" --corpus "$CORPUS"
            --qa-dir "$ROOT_DIR/dataset/QA" --reader-checkpoint "$QA_CHECKPOINT"
            --reader-method "$QA_METHOD" --top-k "$QA_TOP_K"
            --max-source-length "$QA_MAX_SOURCE_LENGTH" --max-target-length "$QA_MAX_TARGET_LENGTH"
            --max-passage-tokens "$QA_MAX_PASSAGE_TOKENS"
            --output "$qa_output"
        )
        if [[ -n "$QA_MODEL_NAME" ]]; then
            QA_ARGS+=(--reader-model-name "$QA_MODEL_NAME")
        fi
        if [[ "$SKIP_BERTSCORE" == "1" ]]; then
            QA_ARGS+=(--skip-bertscore)
        fi
        "$PYTHON" "${QA_ARGS[@]}"
    fi
}

for SEED in "${SEED_ARRAY[@]}"; do
    echo "[2/3] seed=$SEED track=$TRACK methods=${METHOD_ARRAY[*]}"
    for METHOD in "${METHOD_ARRAY[@]}"; do
        case "$METHOD" in
            bm25)
                ;;
            dense)
                train_dense_method dense "$SEED"
                ;;
            hybrid)
                train_dense_method dense "$SEED"
                ;;
            mdr|m3|baleen|mopo)
                train_dense_method "$METHOD" "$SEED"
                ;;
            gmr)
                MODEL_DIR="$OUTPUT_ROOT/models/${TRACK}_gmr_seed${SEED}"
                if [[ ! -f "$MODEL_DIR/COMPLETE" ]]; then
                    "$PYTHON" -m ir.train_gmr \
                        --track "$TRACK" --data-dir "$ROOT_DIR/dataset/IR/experiments" \
                        --output-root "$OUTPUT_ROOT" --model-name "$GMR_MODEL" \
                        --epochs "$GMR_EPOCHS" --max-steps "$MAX_STEPS" \
                        --max-train-examples "$MAX_TRAIN_EXAMPLES" \
                        --batch-size "${GMR_BATCH_SIZE:-4}" \
                        --gradient-accumulation "${GMR_GRADIENT_ACCUMULATION:-4}" \
                        --learning-rate "${GMR_LEARNING_RATE:-3e-4}" \
                        --save-steps "$SAVE_STEPS" --seed "$SEED"
                else
                    echo "Completed GMR model found; retaining its checkpoint."
                fi
                ;;
            ircot)
                ;;
            *)
                echo "Unknown IR method: $METHOD" >&2
                exit 2
                ;;
        esac
        echo "Retrieving dev: $METHOD (seed=$SEED)"
        run_retrieval "$METHOD" dev "$SEED"
        if [[ "$EVALUATE_TEST" == "1" ]]; then
            echo "Retrieving held-out test: $METHOD (seed=$SEED)"
            run_retrieval "$METHOD" test "$SEED"
        fi
    done
done

SUMMARY_SPLITS=(dev)
if [[ "$EVALUATE_TEST" == "1" ]]; then
    SUMMARY_SPLITS+=(test)
fi
SUMMARY_ARGS=(
    -m ir.summarize --output-root "$OUTPUT_ROOT" --track "$TRACK"
    --methods "${METHOD_ARRAY[@]}" --seeds "${SEED_ARRAY[@]}" --splits "${SUMMARY_SPLITS[@]}"
    --search-depth "$SEARCH_DEPTH"
)
"$PYTHON" "${SUMMARY_ARGS[@]}"
if [[ "$EVALUATE_QA" == "1" ]]; then
    "$PYTHON" -m ir.summarize_qa --output-root "$OUTPUT_ROOT" --track "$TRACK" \
        --methods "${METHOD_ARRAY[@]}" --seeds "${SEED_ARRAY[@]}" \
        --splits "${SUMMARY_SPLITS[@]}" --qa-top-k "$QA_TOP_K"
fi
echo "IR experiments completed. Results are under $OUTPUT_ROOT/runs."
