from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import numpy as np


class LDA:

    def __init__(self, n_topics=5, random_state=42):
        self.n_topics = n_topics
        self.random_state = random_state

    def fit(self, target_feature, co_feature):

        def get_vector(lda_model, corpus):

            topic_vectors = lda_model.get_document_topics(
                corpus, minimum_probability=1e-10)
            vector = np.array(
                [topic_vector[1] for topic_vector in topic_vectors])

            return vector

        target_co_sentences = {}
        for target_feature_val, co_feature_val in zip(target_feature,
                                                      co_feature):
            target_co_sentences.setdefault(target_feature_val,
                                           []).append(str(co_feature_val))

        sentences = list(target_co_sentences.values())
        dictionary = Dictionary(sentences)
        corpus = [dictionary.doc2bow(tokens) for tokens in sentences]
        lda = LdaModel(corpus,
                       num_topics=self.n_topics,
                       id2word=dictionary,
                       random_state=self.random_state)
        target_feature_vals = list(target_co_sentences.keys())

        vectors = list(
            map(get_vector, [lda for _ in range(len(corpus))],
                [text for text in corpus]))
        target_feature_val_vectors = {
            target_feature_val: vector
            for target_feature_val, vector in zip(target_feature_vals, vectors)
        }
        return target_feature_val_vectors
