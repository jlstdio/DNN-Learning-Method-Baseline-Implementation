#!/bin/bash
# =============================================================================
# Downstream Task: Pretrained + Random Init 전체 실행 스크립트
#
# 1) Pretrained: 6개 SSL 방법 × 2개 데이터셋 = 12 runs
# 2) Random Init: 2개 데이터셋 = 2 runs  (동일 backbone, method 무관)
# 총 14 runs
# =============================================================================

set -e

GPU=7
MODEL_ID="distilbert/distilbert-base-uncased"
CKPT_DIR="checkpoints"
SAVE_BASE="downstream_task/results"

METHODS=("cmc" "limu_bert" "phymask" "simclr" "ts2vec" "vanilla_mae")
DATASETS=("hhar" "pamap2")

echo "============================================================"
echo " Downstream Task — All experiments"
echo " GPU: ${GPU}  |  Model: ${MODEL_ID}"
echo "============================================================"

# -----------------------------------------------------------------
# 1) Pretrained downstream (6 methods × 2 datasets = 12 runs)
# -----------------------------------------------------------------
echo ""
echo ">>> [Phase 1] Pretrained downstream runs"
echo ""

for METHOD in "${METHODS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        CKPT="${CKPT_DIR}/${METHOD}_distilbert_distilbert_base_uncased_${DATASET}_best.pt"
        SAVE_DIR="${SAVE_BASE}/pretrained_${METHOD}_${DATASET}"

        if [ ! -f "${CKPT}" ]; then
            echo "⚠  Checkpoint not found, skipping: ${CKPT}"
            continue
        fi

        echo "------------------------------------------------------------"
        echo "  Pretrained | Method: ${METHOD} | Dataset: ${DATASET}"
        echo "  Checkpoint: ${CKPT}"
        echo "  Save dir  : ${SAVE_DIR}"
        echo "------------------------------------------------------------"

        CUDA_VISIBLE_DEVICES=${GPU} python downstream_task/train_pretrained.py \
            --checkpoint "${CKPT}" \
            --dataset "${DATASET}" \
            --model_id "${MODEL_ID}" \
            --save_dir "${SAVE_DIR}"

        echo ""
    done
done

# -----------------------------------------------------------------
# 2) Random Init downstream (2 datasets)
# -----------------------------------------------------------------
echo ""
echo ">>> [Phase 2] Random Init downstream runs"
echo ""

for DATASET in "${DATASETS[@]}"; do
    SAVE_DIR="${SAVE_BASE}/random_init_${DATASET}"

    echo "------------------------------------------------------------"
    echo "  Random Init | Dataset: ${DATASET}"
    echo "  Save dir    : ${SAVE_DIR}"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=${GPU} python downstream_task/train_random_init.py \
        --dataset "${DATASET}" \
        --model_id "${MODEL_ID}" \
        --save_dir "${SAVE_DIR}"

    echo ""
done

echo "============================================================"
echo " ✔  All downstream experiments finished!"
echo "    Results saved under: ${SAVE_BASE}/"
echo "============================================================"
