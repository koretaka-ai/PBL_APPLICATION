#!/bin/bash

# --- PATH settings ---
SPM_DATASETS_DIR=../../datasets/small_parallel_enja/sentencepiece
PRE_TRAINED_DIR=../../pre-trained
SRC=en
TRG=ja


# --- Download jparacrawl_base_model ---
mkdir -p $PRE_TRAINED_DIR
pushd $PRE_TRAINED_DIR
if [ -d ./base_model ]; then
    echo "[Info] Pre-trained jparacrawl base model already exists, skipping download"
else
    echo "[Info] Downloading Pre-trained jparacrawl base model"
    wget http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/1.0/pretrained_models/en-ja/base.tar.gz
    tar xzvf base.tar.gz
    rm base.tar.gz
    mv base ./base_model
fi
popd


# --- fairseq-preprocess ---
preprocessed_dir=${SPM_DATASETS_DIR}/fairseq-preprocess
rm -rf $preprocessed_dir
mkdir -p $preprocessed_dir

fairseq-preprocess --source-lang $SRC --target-lang $TRG \
    --trainpref $SPM_DATASETS_DIR/train \
    --validpref $SPM_DATASETS_DIR/dev \
    --testpref $SPM_DATASETS_DIR/test \
    --destdir $preprocessed_dir \
    --srcdict $PRE_TRAINED_DIR/base_model/dict.en.txt --tgtdict $PRE_TRAINED_DIR/base_model/dict.ja.txt
    
