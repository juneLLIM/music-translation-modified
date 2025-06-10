# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

DATE=$(date +%d_%m_%Y)
CODE=src
EXP=Band
OUTPUT=results/${DATE}/${EXP}
EPOCH=328
DECODERS="0 1 2"
DATA=data/preprocessed/day6

echo "Sampling"
python ${CODE}/data_samples.py --data-from-args checkpoints/${EXP}/args.pth --output ${OUTPUT}-py -n 2 --seq 80000 --data ${DATA}

echo "Generating"
python ${CODE}/run_on_files.py --files ${OUTPUT}-py --batch-size 2 --checkpoint checkpoints/${EXP}/lastmodel_epoch_${EPOCH} --output-next-to-orig --decoders ${DECODERS} --py
