import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from category_encoders.hashing import HashingEncoder
from lib import data_handler
from lib import evaluator

# カラム名の定義
ground_truth_column = ['is_click']
integer_columns = ['integer_' + str(i + 1) for i in range(13)]
categorical_columns = ['categorical_' + str(i + 1) for i in range(26)]

# 入力データパス
input_data_path = 'data/dac_sample.txt'

# テストデータの割合
test_rate = 0.2

# ハッシング後の次元数
n_hash_dims = 128


def train_test_fh():

    # データ読み込み
    df = pd.read_csv('data/dac_sample.txt', sep='\t', header=None)
    df.columns = ground_truth_column + integer_columns + categorical_columns

    df_train, df_test = data_handler.train_test_split(df, test_rate)

    # サンプリング
    # サンプリング後のインデックスが欲しいのでラベル以外はダミーデータを与える
    # 圧倒的に高速
    sampled_indicies = data_handler.under_sampling(
        X=np.zeros((len(df_train), 1), dtype=np.uint8),
        y=df_train[ground_truth_column].values.astype(int))
    df_train = df_train.query('index in @sampled_indicies')

    # NULL値の処理
    df_train = data_handler.fillna_integer_feature(df_train, integer_columns)
    df_train = data_handler.fillna_categorical_feature(df_train,
                                                       categorical_columns)

    # Hashing
    hasher = HashingEncoder(cols=categorical_columns, n_components=n_hash_dims)
    df_train = hasher.fit_transform(df_train)
    print(df_train.head())

    # 学習
    X_train = np.array(df_train.drop(ground_truth_column, axis=1).values)
    y_train = np.array(df_train[ground_truth_column].values)
    model = LogisticRegression(random_state=42, solver='lbfgs')
    model.fit(X_train, y_train)

    # テストデータの処理
    df_test = data_handler.fillna_integer_feature(df_test, integer_columns)
    df_test = data_handler.fillna_categorical_feature(df_test,
                                                      categorical_columns)
    df_test = hasher.transform(df_test)

    # 予測
    X_test = np.array(df_test.drop(ground_truth_column, axis=1).values)
    y_test = np.array(df_test[ground_truth_column].values)
    y_proba = model.predict_proba(X_test)

    # 評価
    logloss = evaluator.logloss(y_test, y_proba[:, 1])
    print(logloss)


if __name__ == '__main__':

    train_test_fh()
