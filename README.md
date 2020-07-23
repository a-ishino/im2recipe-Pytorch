# im2recipe-Pytorch
Learning Cross-modal Embeddings for Cooking Recipes and Food Images (CVPR2017) の[公開コード](https://github.com/torralba-lab/im2recipe-Pytorch)を改良したもの

## 1-1. Train用データの用意
`layer1.json` （`ingredients`, `partition`, `title`, `id`, `instructions`）、 <br>
`layer2.json` （`id`、 `images`）、<br>
`det_ingrs.json` （`valid`、 `id`、 `ingredients`） および画像データを用意する。<br>

```
data_dir
├── layers
│   ├── layer1.json
│   ├── layer2.json
│   └── det_ingrs.json
├── images
│   ├── train
│   ├── test
│   └── val
├── food101_classes_renamed.txt
└── remove1M.txt
```

```
im2recipe-Pytorch
├── data
│   └── tag
├── logs
│   └── tag
├── results
│   └── tag
├── scripts
│   └── _word2vec
└── snapshots
    └── tag
```

## 1-2. Installation
```
conda create --name recipe1m python=3.7
sorce activate recipe1m
conda install pytorch=1.4.0
conda install gxx_linux-64 # これめっちゃ大事
pip install torchwordemb
conda install scipy=1.1.0

conda install simplejson
conda install tqdm
conda install pillow
conda install nltk
conda install word2vec

conda install numpy
conda install scikit-learn
conda install transformers=2 -c conda-forge

conda install python-lmdb
conda install torchfile -c conda-forge

conda install torchvision
conda install tensorboard
```

## 1-3. 小さめの入力データを用意する。
```
!python mk_small_data.py -data_path=/data_dir/layers/ -out_path=../data -tag=tag
```

## 2-1. word2vecのtrain
料理手順 instructions について、token化した上でword2vecをtrainする。`vocab.bin` を生成。 <br>

### 2-1-1. tokenize
`layer1.json`　の instructions を token化。 `tokenized_insts_train.txt` を生成。
- INPUT : `layer1.json`, `det_ingrs.json` <br>
- OUTPUT : `tokenized_insts_train.txt`, `tokenized_insts_val.txt`, `tokenized_insts_test.txt` <br>
```
!python tokenize_instructions.py -data_path=../data -tag=tag
```

### 2-1-2. word2vecのビルド
```
!cd ~/workspace/joint-embedding/im2recipe-Pytorch/scripts
!wget https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip
!unzip source-archive.zip
!mv word2vec/ _word2vec/  # rename
!cd _word2vec/trunk/
!make word2vec
!cd ../../
```

### 2-1-3. vocab.bin の生成
`tokenized_insts_train.txt` を用いてword2vecのtrain。 `vocab.bin` を生成。
- INPUT : `tokenized_insts_train.txt`, `det_ingrs.json` <br>
- OUTPUT : `tokenized_insts_train.txt`, `tokenized_insts_val.txt`, `tokenized_insts_test.txt` <br>
```
!./_word2vec/trunk/word2vec -hs 1 -negative 0 -window 10 -cbow 0 -iter 10 -size 300 -binary 1 -min-count 20 -threads 20 -train ../data/tag/insts/tokenized_insts_train.txt -output ../data/tag/vocab.bin
```

## 2-2. 料理 title の class 分類？
この部分使ってなくね？ → mk_dataset.py で使ってた
### 2-2-1. bigram の生成
料理名 title について、頻度順のbi-gram (`bigram.pkl`） と 料理名一覧（`titles.txt`） を生成 <br>
- INPUT : `layer1.json`, `layer2.json`, `det_ingrs.json` <br>
- OUTPUT : `bigrams.pkl`, `titles.txt` <br>
```
!python bigrams.py --crtbgrs -data_path=../data -tag=tag
```

### 2-2-2. classes.pkl の生成
bigramの結果からクラスを抽出
- INPUT : `vocab.txt`, `layer2.json`, `det_ingrs.json` <br>
- OUTPUT : `bigrams.pkl`, `titles.txt` <br>
```
!python bigrams.py --nocrtbgrs -data_path=../data -tag=tag -f101_cats=data_dir/food101_classes_renamed.txt
```

## 2-3. instructions の embedding
`layer1.json` の `instructions` について skip-thoughts でベクトル化したもの = `skip-instructions` を生成するのだが、Pytorchのコードが公開されていないため、BERTによるembeddingに置き換える。<br>
GPUサーバにログインして以下を実行。
```
### Run on GPU
python bert-embedding.py -data_path=../data -tag=tag
```
## 2-4. データセットをLMDB形式にまとめる
mk_dataset.py は parameter 参照時に ../args.py を見ているので注意  
```
### Run on GPU
python mk_dataset.py -emb_type=bert -data_path=../data/ -tag=tag -remove1m=data_dir/remove1M.txt
```

## 3. Training
公式のLMDBを用いる場合は、inst_emb=st を指定する。 <br>
BERTでembedした結果のLMDBを用いる場合は、inst_emb=bertを指定する(default)
```
cd workspace/joint-embedding/im2recipe-Pytorch/
python train.py \
-img_path=data_dir/images/ \
-data_path=data \
-tag=tag \
-n_samples=200 \
-medr=100 
```

## 4. Test
train したモデルを用いて、testデータをembed
```
python test.py  \
-img_path=data_dir/images/ \
-data_path=data \
-tag=tag \
-model_path=snapshots/tag/yyyy-mm-dd_HH:MM/model_nXXX_eYYY_v-ZZ.ZZZ.pth.tar
```