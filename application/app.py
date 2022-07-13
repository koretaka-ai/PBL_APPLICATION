from flask import Flask, render_template, request
import os, subprocess, sys

app = Flask(__name__)

# --- PATH settings ---
model = "../translate/results/SEED33/checkpoints/checkpoint_best.pt"
spm_en = "../translate/pre-trained/enja_spm_models/spm.en.nopretok.model"
spm_ja = "../translate/pre-trained/enja_spm_models/spm.ja.nopretok.model"

datasets = "./datasets/"
input = datasets + "test.en"
input_spm = datasets + "sentencepiece/spm.en"
output = "./result/result"
dest_dir = "../translate/datasets/small_parallel_enja/sentencepiece/fairseq-preprocess"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['post'])
def trans():
    en = request.form['en']
    ja = en
    answer = ''
    if not en:
        answer = ''
        return render_template('index.html', ja=answer, en=en)
    # 翻訳したい文の書き込み
    with open(input, "w") as f:
        f.write(en)
    
    # spmの前処理
    spm_preprocess = "spm_encode --model={} --output_format=piece < {} > {}".format(spm_en, input, input_spm)
    subprocess.run(spm_preprocess, shell=True)
    
    # データの翻訳
    fairseq_interactive = "CUDA_VISIBLE_DEVICES=0,1,2,3 /home/koretaka/anaconda3/envs/trans-en-ja/bin/fairseq-interactive {} \
    --input {} \
    --path {} \
    --batch-size 64 --buffer-size 1024 \
    --nbest 1 --beam 5 \
    > {}.txt".format(dest_dir, input_spm, model, output)
    subprocess.run(fairseq_interactive, shell=True)
    
    # データの抽出
    result_grep = "grep '^H-' {}.txt | sort -V | cut -f3 > {}.ja".format(output, output)
    subprocess.run(result_grep, shell=True)

    detokenize = "cat {}.ja | sed 's/<<unk>>/<unk>/g' | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^[ \t]*//g' | sed 's/ //g' > {}.detok.ja".format(output, output)
    subprocess.run(detokenize, shell=True)

    with open(output + ".detok.ja", "r") as file:
        lines = file.readlines()
        s_lines = [line.strip() for line in lines]
        for line in s_lines:
            answer = line
        
    return render_template('index.html', ja=answer, en=en)

if __name__ == "__main__":
    app.run(debug=True)

# uWSGIはデフォルトでapplicationを探すためapplicationのaliasを書く必要があるらしい
# application = app
