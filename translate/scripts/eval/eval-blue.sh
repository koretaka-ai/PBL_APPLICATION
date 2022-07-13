#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../datasets/small_parallel_enja
RESULT_DIR=../../results
orig_file=$DATASETS_DIR/test.ja

EXP_NAME=RNN
GPU_ID=0
MODEL=checkpoint_best.pt

SPM_JA=../../pre-trained/enja_spm_models/spm.ja.nopretok.model

while getopts d:g:hm:n:r: OPT
do
    case $OPT in
	d ) DATASETS_DIR=$OPTARG
	    ;;
	g ) GPU_ID=$OPTARG
	    ;;
	h ) echo "Usage: $0 [-d DATASETS_DIR] [-g GPU_ID] [-m MODEL] [-n EXP_NAME] [-r RESULT_DIR]" 1>&2
	    exit 1
	    ;;
	m ) MODEL=$OPTARG
	    ;;
	n ) EXP_NAME=$OPTARG
	    ;;
	r ) RESULT_DIR=$OPTARG
	    ;;
    esac
done

preprocessed_dir=$DATASETS_DIR/sentencepiece/fairseq-preprocess
model=$RESULT_DIR/$EXP_NAME/checkpoints/$MODEL
output_dir=$RESULT_DIR/$EXP_NAME/result
result=$output_dir/result
log=$output_dir/BLEU.log

if [ -e $model ]; then
    mkdir -p $output_dir
    CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-generate $preprocessed_dir \
			--path $model \
			--beam 5 --max-sentences 5 \
			> ${result}.txt

    grep ^H ${result}.txt | sort -V | cut -f 3- > ${result}.sys.temp
    spm_decode --model=$SPM_JA --input_format=piece < ${result}.sys.temp > ${result}.sys

    cat $orig_file | mecab -Owakati > ${result}.tok.ref
    cat ${result}.sys | mecab -Owakati > ${result}.tok.sys

    fairseq-score --sys ${result}.tok.sys --ref ${result}.tok.ref \
		  2>&1 | tee -a $log
    
else
    echo "[Info] this training data no exists"
fi
