import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from category_encoders.hashing import HashingEncoder
from lib import data_handler
from lib import evaluator
from lib.topic_model import LDA

# カラム名の定義
ground_truth_column = ['is_click']
integer_columns = ['integer_' + str(i + 1) for i in range(13)]
categorical_columns = ['categorical_' + str(i + 1) for i in range(26)]

# 入力データパス
input_data_path = 'data/dac_sample.txt'

# テストデータの割合
test_rate = 0.2

# 各カテゴリ変数ごとにLDAの特徴量を作成する特徴量数
n_lda_features = 5

# トピック数
n_topics = 5


def convert_lda_feature(key_vectors, X, n_lda_dims, categorical_columns):

    n_dims = n_lda_dims * len(categorical_columns)
    n_samples = X.shape[0]

    X_lda = np.zeros((n_samples, n_dims))

    for sample_idx in range(n_samples):
        latter_idx = 0
        for feature_idx in range(len(categorical_columns)):
            try:
                vector = key_vectors[(categorical_columns[feature_idx],
                                      X[sample_idx, feature_idx])]
            except KeyError:
                vector = np.ones(n_lda_dims) * -1.0
            former_idx = latter_idx
            latter_idx += n_lda_dims
            X_lda[sample_idx, former_idx:latter_idx] = vector

    return X_lda


def train_test_lda():

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

    # LDA
    n_lda_dims = n_lda_features * n_topics
    X_train_categorical = np.array(df_train[categorical_columns].values)
    key_vectors = {}
    for seq_idx, target_column in enumerate(categorical_columns):
        counter = 0
        choiced_indicies = [
            i for i in range(len(categorical_columns)) if i != seq_idx
        ]
        co_indicies = np.random.choice(choiced_indicies,
                                       n_lda_features,
                                       replace=False)
        for cat_idx in co_indicies:
            lda = LDA(n_topics=n_topics)
            target_feature_val_vectors = lda.fit(
                X_train_categorical[:, seq_idx],
                X_train_categorical[:, cat_idx])
            if counter == 0:
                key_vectors.update({
                    (target_column, val): vector
                    for val, vector in target_feature_val_vectors.items()
                })
            else:
                key_vectors.update({
                    (target_column, val): np.hstack(
                        (key_vectors[(target_column, val)], vector))
                    for val, vector in target_feature_val_vectors.items()
                })

            counter += 1

    X_train_lda = convert_lda_feature(key_vectors, X_train_categorical,
                                      n_lda_dims, categorical_columns)
    print(X_train_lda)
    print(X_train_lda.shape)

    # 学習
    X_train_integer = np.array(
        df_train.drop(ground_truth_column + categorical_columns, axis=1).values)
    X_train = np.hstack((X_train_lda, X_train_integer))
    y_train = np.array(df_train[ground_truth_column].values)
    model = LogisticRegression(random_state=42, solver='lbfgs')
    model.fit(X_train, y_train)

    # テストデータの処理
    df_test = data_handler.fillna_integer_feature(df_test, integer_columns)
    df_test = data_handler.fillna_categorical_feature(df_test,
                                                      categorical_columns)

    # LDA
    X_test_categorical = np.array(df_test[categorical_columns].values)
    X_test_lda = convert_lda_feature(key_vectors, X_test_categorical,
                                     n_lda_dims, categorical_columns)
    print(X_test_lda)
    print(X_test_lda.shape)

    # 予測
    X_test_integer = np.array(
        df_test.drop(ground_truth_column + categorical_columns, axis=1).values)
    X_test = np.hstack((X_test_lda, X_test_integer))
    y_test = np.array(df_test[ground_truth_column].values)
    y_test = y_test.reshape(y_test.shape[0])
    y_proba = model.predict_proba(X_test)

    # 評価
    logloss = evaluator.logloss(y_test, y_proba[:, 1])
    print(logloss)


if __name__ == '__main__':

    train_test_lda()
