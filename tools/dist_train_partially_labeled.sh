#!/usr/bin/env bash
set -x

CONFIG=configs/PseCo/PseCo_swin_transformer_custom.py   
work_dir=./work_dir/          # define your experiment path here

FOLD=1
PERCENT=10

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

export unsup_start_iter=200
export unsup_warmup_iter=100

python -m torch.distributed.launch --nproc_per_node=1 --master_port=${PORT:-29500} \
    $(dirname "$0")/train.py $CONFIG --work-dir $work_dir --launcher=pytorch \
    --cfg-options fold=${FOLD} \
                  percent=${PERCENT} \


# For 1% and 2% labelling ratios, shorter training shedule can 
# alleviate over-fitting. Therefore, 
# runner.max_iters=90000 \
# lr_config.step=\[60000\]