from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import numpy as np


class LDA:

    def __init__(self,
                 co_features,
                 n_topics=5,
                 n_lda_features=5,
                 random_state=42):
        self.co_features = co_features
        self.n_topics = n_topics
        self.n_lda_features = n_lda_features
        self.n_lda_dims = n_topics * n_lda_features
        self.random_state = random_state
        self.unk_vectors = np.ones(self.n_lda_dims) * -1.0
        self.key_vectors = {}

    def fit(self, df):

        def get_vector(lda_model, corpus):

            topic_vectors = lda_model.get_document_topics(
                corpus, minimum_probability=1e-10)
            vector = np.array(
                [topic_vector[1] for topic_vector in topic_vectors])

            return vector

        key_vectors = {}
        # 共起行列を作成する特徴量ペアごとに処理
        for co_feature in self.co_features:
            target_feature = np.array(df[co_feature[0]].values)
            counterpart_feature = np.array(df[co_feature[1]].values)
            target_co_sentences = {}
            # 共起行列を作成
            for target_feature_val, counterpart_feature_val in zip(
                    target_feature, counterpart_feature):
                target_co_sentences.setdefault(target_feature_val, []).append(
                    str(counterpart_feature_val))

            sentences = list(target_co_sentences.values())
            dictionary = Dictionary(sentences)
            corpus = [dictionary.doc2bow(tokens) for tokens in sentences]

            # LDA
            lda = LdaModel(corpus,
                           num_topics=self.n_topics,
                           id2word=dictionary,
                           random_state=self.random_state)

            # ターゲットの特徴量の値ごとにトピックのベクトルを取得
            vectors = list(
                map(get_vector, [lda for _ in range(len(corpus))],
                    [text for text in corpus]))

            # (ターゲットの特徴量、ターゲットの特徴量の値)をkey、トピックのベクトルをvalueとして辞書に格納
            target_feature_vals = list(target_co_sentences.keys())
            target_column = co_feature[0]
            for target_feature_val, vector in zip(target_feature_vals, vectors):
                key = (target_column, target_feature_val)
                if key not in key_vectors:
                    key_vectors.update({key: vector})
                else:
                    key_vectors.update(
                        {key: np.hstack((key_vectors[key], vector))})

        self.key_vectors = key_vectors

    def transform(self, df, categorical_columns):

        if len(self.key_vectors) == 0:
            raise Exception('Need to train before transform')

        n_dims = self.n_lda_dims * len(categorical_columns)
        n_samples = len(df)

        X = np.array(df[categorical_columns].values)
        X_lda = np.zeros((n_samples, n_dims))

        for sample_idx in range(n_samples):
            latter_idx = 0
            for feature_idx in range(len(categorical_columns)):
                try:
                    vector = self.key_vectors[(categorical_columns[feature_idx],
                                               X[sample_idx, feature_idx])]
                # 学習時には無かった未知のキーの場合
                except KeyError:
                    vector = self.unk_vectors
                former_idx = latter_idx
                latter_idx += self.n_lda_dims
                X_lda[sample_idx, former_idx:latter_idx] = vector

        return X_lda
