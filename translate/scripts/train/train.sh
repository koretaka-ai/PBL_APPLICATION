#!/bin/bash

# --- PATH settings ---
PRETRAINED_MODEL=../../pre-trained/base_model/base.pretrain.pt
RESULT_DIR=../../results

EXP_NAME=SEED11
GPU_ID=0
SEED=11

while getopts g:hn:r:s: OPT
do
    case $OPT in
    g ) GPU_ID=$OPTARG
        ;;
    h ) echo "Usage: $0 [-g GPU_ID] [-n EXP_NAME] [-r RESULT_DIR] [-s SEED]" 1>&2
        exit 1
        ;;
    n ) EXP_NAME=$OPTARG
        ;;
    r ) RESULT_DIR=$OPTARG
        ;;
    s ) SEED=$OPTARG
        ;;
    esac
done


# --- fairseq train ---

preprocessed_dir=../../datasets/small_parallel_enja/sentencepiece/fairseq-preprocess
save_dir=${RESULT_DIR}/${EXP_NAME}
rm -rf $save_dir
mkdir -p $save_dir

echo Training server name : `hostname` > ${save_dir}/train.log
CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-train $preprocessed_dir \
		    --restore-file $PRETRAINED_MODEL \
		    --arch transformer --optimizer adam \
		    --source-lang en --target-lang ja \
		    --save-dir ${save_dir}/checkpoints \
		    --clip-norm 0.0 \
		    --seed $SEED \
		    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
		    --warmup-updates 4000 --lr 3e-5 --dropout 0.1 \
		    --weight-decay 0.0 \
		    --criterion label_smoothed_cross_entropy \
		    --label-smoothing 0.1 \
		    --max-tokens 5000 --no-epoch-checkpoints \
		    --reset-optimizer \
		    --max-epoch 100 --reset-dataloader \
		    --patience 5 \
		    >> ${save_dir}/train.log
