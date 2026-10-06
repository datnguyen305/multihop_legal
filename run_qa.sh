#!/usr/bin/env bash

# One-click launcher for the abstractive QA experiment.
# Override any setting when needed, for example:
#   METHOD=msg MODEL_NAME=google/mt5-small MAX_STEPS=1 ./run_qa.sh

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
METHOD="${METHOD:-pathfid}"
TRACK="${TRACK:-true_multihop}"
MODEL_NAME="${MODEL_NAME:-google/mt5-base}"
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:-0}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-0}"
SAVE_STEPS="${SAVE_STEPS:-50}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-1024}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-1024}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/qa}"
PREPARE_DATA="${PREPARE_DATA:-1}"
EVALUATE="${EVALUATE:-1}"
SKIP_BERTSCORE="${SKIP_BERTSCORE:-1}"
DATA_DIR="$ROOT_DIR/dataset/QA/abstractive"
CORPUS_DIR="$ROOT_DIR/dataset/IR/structured"
RUN_DIR="$OUTPUT_DIR/${TRACK}_${METHOD}_seed42"

echo "=== Multihop legal QA ==="
echo "method=$METHOD track=$TRACK model=$MODEL_NAME"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Không tìm thấy Python: $PYTHON" >&2
    exit 1
fi

"$PYTHON" - <<'PY'
import importlib.util
missing = [name for name in ("torch", "transformers", "sentencepiece")
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Thiếu package: " + ", ".join(missing) +
        ". Hãy chạy: python -m pip install -r requirements-qa.txt"
    )
PY

if [[ "$PREPARE_DATA" == "1" ]]; then
    if [[ ! -f "$DATA_DIR/manifest.json" || \
          ! -f "$DATA_DIR/$TRACK/train.jsonl" || \
          ! -f "$DATA_DIR/$TRACK/dev.jsonl" || \
          ! -f "$CORPUS_DIR/train_corpus.jsonl" || \
          ! -f "$CORPUS_DIR/dev_corpus.jsonl" ]]; then
        echo "[1/3] Đang chuẩn bị dữ liệu QA..."
        "$PYTHON" -m qa.prepare_abstractive
    else
        echo "[1/3] Dữ liệu QA đã tồn tại, bỏ qua bước prepare."
    fi
else
    echo "[1/3] Bỏ qua bước prepare theo PREPARE_DATA=$PREPARE_DATA."
fi

TRAIN_ARGS=(
    -m qa.train
    --method "$METHOD"
    --track "$TRACK"
    --model-name "$MODEL_NAME"
    --epochs "$EPOCHS"
    --max-source-length "$MAX_SOURCE_LENGTH"
    --max-target-length "$MAX_TARGET_LENGTH"
    --save-steps "$SAVE_STEPS"
    --output-dir "$OUTPUT_DIR"
)
if [[ "$MAX_STEPS" != "0" ]]; then
    TRAIN_ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ "$MAX_TRAIN_EXAMPLES" != "0" ]]; then
    TRAIN_ARGS+=(--max-train-examples "$MAX_TRAIN_EXAMPLES")
fi

echo "[2/3] Bắt đầu training..."
if [[ -f "$RUN_DIR/COMPLETE" ]]; then
    echo "Đã tìm thấy COMPLETE marker tại $RUN_DIR; bỏ qua training."
elif [[ -f "$RUN_DIR/latest_checkpoint.txt" ]]; then
    echo "Đã tìm thấy checkpoint gần nhất: $(<"$RUN_DIR/latest_checkpoint.txt")"
    echo "Trainer sẽ tự động resume từ checkpoint này."
    "$PYTHON" "${TRAIN_ARGS[@]}"
else
    echo "Chưa có checkpoint; khởi tạo model từ $MODEL_NAME."
    "$PYTHON" "${TRAIN_ARGS[@]}"
fi

CHECKPOINT="$RUN_DIR"
if [[ "$EVALUATE" == "1" ]]; then
    echo "[3/3] Đánh giá trên dev..."
    EVAL_ARGS=(
        -m qa.evaluate
        --checkpoint "$CHECKPOINT"
        --method "$METHOD"
        --model-name "$MODEL_NAME"
        --data "$DATA_DIR/$TRACK/dev.jsonl"
        --corpus "$CORPUS_DIR/dev_corpus.jsonl"
        --max-source-length "$MAX_SOURCE_LENGTH"
        --max-target-length "$MAX_TARGET_LENGTH"
        --output "$CHECKPOINT/dev_metrics.json"
    )
    if [[ "$SKIP_BERTSCORE" == "1" ]]; then
        EVAL_ARGS+=(--skip-bertscore)
    fi
    "$PYTHON" "${EVAL_ARGS[@]}"
else
    echo "[3/3] Bỏ qua evaluation theo EVALUATE=$EVALUATE."
fi

echo
echo "Hoàn tất. Checkpoint: $CHECKPOINT"
