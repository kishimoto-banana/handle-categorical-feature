# handle-categorical-feature

## 概要
カテゴリ変数を扱う実験リポジトリ
- feature hashing
- LDA-based feature extraction

## 前提
- Python 3.x

## セットアップ
- プロジェクトディレクトリに移動
```bash
cd $projectRoot
pip install -r requirements.txt
```

## 実行
```bash
# feature hashing
python feature_hashing.py

# LDA
python lda.py
```

## 構成
```
├── data
│   └── dac_sample.txt   <- 入力データ
├── exam_data.py         <- データの調査スクリプト
├── feature_hashing.p    <- feature hashingの実行スクリプト
├── lda.py               <- LDA-based feature extractionの実行スクリプト
├── lib 
│   ├── data_handler.py  <- データ処理のモジュール
│   ├── evaluator.py     <- 評価のモジュール
│   └── topic_model.py   <- LDAのモジュール
└── requiements.txt
```
