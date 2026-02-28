# CUDA_VISIBLE_DEVICES=6 python train_ts2vec.py --dataset pamap2 --model_id distilbert/distilbert-base-uncased
# CUDA_VISIBLE_DEVICES=6 python train_ts2vec.py --dataset hhar --model_id distilbert/distilbert-base-uncased

CUDA_VISIBLE_DEVICES=7 python train_ts2vec.py --dataset pamap2 --model_id microsoft/swinv2-tiny-patch4-window8-256
CUDA_VISIBLE_DEVICES=7 python train_ts2vec.py --dataset hhar --model_id microsoft/swinv2-tiny-patch4-window8-256

