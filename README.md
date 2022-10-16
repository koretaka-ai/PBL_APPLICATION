# PBL memo

# 翻訳アプリに使うモデルの作成
- small_parallel_enjaをjaparacrawl事前学習済みモデル（ベースモデル）でfinetuiningする
- python 環境 3.8.10
## git clone
~~~
git clone https://github.com/koretaka-ai/PBL_APPLICATION.git
cd PBL_APPLICATION
~~~
## Anacondaが入っていなかったら導入
- 詳しく説明しているサイト：https://hana-shin.hatenablog.com/entry/2022/02/12/203642
~~~
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
~~~
## 仮想環境構築
~~~
conda create -n PBL_t python=3.8.10 -y
conda activate PBL_t
~~~
## 必要なライブラリのインストール
- fairseq install 
~~~
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ..
~~~
- sentencepiece install
~~~
git clone https://github.com/google/sentencepiece.git 
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v
cd ../..
~~~
- vcpkg install
~~~
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
./vcpkg install sentencepiece
cd ..
~~~
- flask install
~~~
pip install Flask
~~~
- sentencepiece install
~~~
pip install sentencepiece
~~~
## データの準備
- データの前処理 small_parallel_enjaのdetokenizeとjparacrawl pre-trained-modelを使うためにdownloadしたspm modelでtokenizeする
~~~
pushd translate/scripts/preprocess
bash preprocess.sh
popd
~~~
## モデルの訓練
- fairseqの前処理と訓練
- GPUを使う際には `nvidia-smi` を使用してGPUがあいているか確認
~~~ 
pushd translate/scripts/train
bash fairseq_preprocess.sh
bash train.sh -n ${実験名} -g ${GPUのID} -s ${SEED値}
# example
# bash train.sh -n SEED33 -g 0 -s 33 
popd
~~~
# アプリケーションの実行・確認
## アプリの起動
- 127.0.0.1:5000 に翻訳アプリが表示される
- app.py の model の path を適切なものに書き換えてください
~~~
cd application
python app.py
~~~
# ngrok を利用して Flask で立ち上げたアプリケーションを外部に公開する
1. ngrok の公式サイトから linux version のものを install して 解凍
2. nginx が入ってなかったら install 
3. 以下のコマンドを入力し, localhost:5000 を外部に公開する
~~~
cd [ngrokをインストールして解凍したディレクトリ]
./ngrok http 5000
python app.py
~~~
