#!/bin/bash

# -- PATH settings ---
DATASETS_DIR=../../datasets
RESOURCES_DIR=../../resources
PRE_TRAINED_DIR=../../pre-trained
SPM_EN=$PRE_TRAINED_DIR/enja_spm_models/spm.en.nopretok.model
SPM_JA=$PRE_TRAINED_DIR/enja_spm_models/spm.ja.nopretok.model


# --- Download sentencepiece model ---
pushd $RESOURCES_DIR
if [ -d .small_parallel_enja ]; then
    echo "[Info] small_parallel_enja datasets already exists, skipping download"
else
    echo "[Info] Downloading small_parallel_enja datasets"
    git clone "https://github.com/odashi/small_parallel_enja.git"
fi
popd


# --- Download sentencepiece model ---
pushd $PRE_TRAINED_DIR
if [ -d ./enja_spm_models ]; then
    echo "[Info] jparacral sentencepiece model already exists, skipping download"
else
    echo "[Info] Downloading jparacrawl sentencepiece model"
    wget http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/1.0/spm_models/en-ja_spm.tar.gz
    tar xzvf en-ja_spm.tar.gz
    rm en-ja_spm.tar.gz
fi
popd


python detokenize.py


# --- tokenize small_parallel_enja dataset ---
pushd $DATASETS_DIR/small_parallel_enja
if [ -d ./sentencepiece ]; then
    echo "[Info] tokenized datasets already exists"
else
    save_dir=$DATASETS_DIR/small_parallel_enja/sentencepiece
    datasets_dir=$DATASETS_DIR/small_parallel_enja
    mkdir -p $save_dir    
    echo "[Info] tokenizing ... "
    spm_encode --model=$SPM_EN --output_format=piece < $datasets_dir/train.en > $save_dir/train.en
    spm_encode --model=$SPM_EN --output_format=piece < $datasets_dir/dev.en   > $save_dir/dev.en
    spm_encode --model=$SPM_EN --output_format=piece < $datasets_dir/test.en  > $save_dir/test.en
    spm_encode --model=$SPM_JA --output_format=piece < $datasets_dir/train.ja > $save_dir/train.ja
    spm_encode --model=$SPM_JA --output_format=piece < $datasets_dir/dev.ja   > $save_dir/dev.ja
    spm_encode --model=$SPM_JA --output_format=piece < $datasets_dir/test.ja  > $save_dir/test.ja
fi
popd
