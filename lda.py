import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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
    co_features = []
    for target_column in categorical_columns:
        candidate_columns = [
            column for column in categorical_columns if column != target_column
        ]
        counterpart_columns = np.random.choice(candidate_columns,
                                               n_lda_features,
                                               replace=False)
        co_features.extend([(target_column, counterpart_column)
                            for counterpart_column in counterpart_columns])

    lda = LDA(co_features, n_topics, n_lda_features)
    lda.fit(df_train)
    X_train_lda = lda.transform(df_train, categorical_columns)

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
    X_test_lda = lda.transform(df_test, categorical_columns)
    print(X_test_lda)
    print(X_test_lda.shape)

    # 予測
    X_test_integer = np.array(
        df_test.drop(ground_truth_column + categorical_columns, axis=1).values)
    X_test = np.hstack((X_test_lda, X_test_integer))
    y_test = np.array(df_test[ground_truth_column].values)
    y_proba = model.predict_proba(X_test)

    # 評価
    logloss = evaluator.logloss(y_test, y_proba[:, 1])
    print(logloss)


if __name__ == '__main__':

    train_test_lda()
