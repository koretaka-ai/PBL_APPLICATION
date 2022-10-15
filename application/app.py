import os
import shutil
from flask import Flask, render_template, request
from fairseq.models.transformer import TransformerModel
import sentencepiece as spm

app = Flask(__name__)

# --- PATH settings ---
preprocess_dir = "../translate/datasets/small_parallel_enja/sentencepiece/fairseq-preprocess"
model_dir = "../translate/results/SEED33/checkpoints"
model_name = "checkpoint_best.pt"
spm_en = "../translate/pre-trained/enja_spm_models/spm.en.nopretok.model"

if os.path.isfile(f'{model_dir}/dict.en.txt') == False:
    shutil.copy(f'{preprocess_dir}/dict.en.txt', f'{model_dir}')

if os.path.isfile(f'{model_dir}/dict.ja.txt') == False:
    shutil.copy(f'{preprocess_dir}/dict.ja.txt', f'{model_dir}')

model = TransformerModel.from_pretrained(model_dir, model_name)
spm_model = spm.SentencePieceProcessor(model_file=spm_en)

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

    # spmの前処理
    spm_en = spm_model.encode(en, out_type=str)
    spm_en = ' '.join(spm_en)

    # 翻訳
    spm_ja = model.translate(spm_en)
    ja = spm_ja.replace('▁', '').replace(' ', '')

    return render_template('index.html', ja=ja, en=en)

if __name__ == "__main__":
    app.run()
