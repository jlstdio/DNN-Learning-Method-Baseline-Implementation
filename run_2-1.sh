# CUDA_VISIBLE_DEVICES=7 python train_limu_bert.py --dataset pamap2 --model_id distilbert/distilbert-base-uncased
# CUDA_VISIBLE_DEVICES=7 python train_limu_bert.py --dataset hhar --model_id distilbert/distilbert-base-uncased

CUDA_VISIBLE_DEVICES=7 python train_limu_bert.py --dataset pamap2 --model_id microsoft/swinv2-tiny-patch4-window8-256
CUDA_VISIBLE_DEVICES=7 python train_limu_bert.py --dataset hhar --model_id microsoft/swinv2-tiny-patch4-window8-256