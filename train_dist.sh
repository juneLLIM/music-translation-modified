# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

set -e -x

CODE=src
DATA=data/preprocessed
EXP=Band
MASTER_PORT=29500

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=${MASTER_PORT} \
    ${CODE}/train.py \
    --data ${DATA}/day6 \
    ${DATA}/silicagel \
    ${DATA}/carthegarden \
    ${DATA}/jannabi \
    ${DATA}/lucy \
    ${DATA}/thornapple \
    --checkpoint checkpoints/pretrained_musicnet/bestmodel_0.pth \
    --per-epoch \
    --batch-size 8 \
    --lr-decay 0.995 \
    --epochs 100 \
    --epoch-len 1000 \
    --num-workers 5 \
    --encoder-lr 1e-4 \
    --decoder-lr 1e-3 \
    --discriminator-lr 1e-4 \
    --seq-len 12000 \
    --d-lambda 1e-2 \
    --expName ${EXP} \
    --latent-d 64 \
    --layers 14 \
    --blocks 4 \
    --data-aug \
    --grad-clip 1
