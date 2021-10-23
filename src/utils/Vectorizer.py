import copy
import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict, Counter


class Vectorizer(ABC):
    def __init__(self, all_tokens, lexicon):
        self.all_tokens = all_tokens
        self.lexicon = lexicon

    @abstractmethod
    def get_vectors(self):
        pass


class TFIDFVectorizer(Vectorizer):
    def get_vectors(self):
        zero_vector = OrderedDict((token, 0) for token in self.lexicon)

        tf_idf = []
        idf_mapping = {}

        num_docs = len(self.all_tokens)
        for token in self.lexicon:
            num_docs_containing_token = 0
            for text_tokens in self.all_tokens:
                if token in text_tokens:
                    num_docs_containing_token += 1

            if num_docs_containing_token == 0:
                idf_mapping[token] = 1
            else:
                idf_mapping[token] = num_docs / num_docs_containing_token

        for text_tokens in self.all_tokens:
            vector = copy.copy(zero_vector)
            token_counts = Counter(text_tokens)
            for token, value in token_counts.items():
                tf = value / len(self.lexicon)
                idf = idf_mapping.get(token, 1)
                vector[token] = tf * idf

            tf_idf.append(vector)

        return [list(vector.values()) for vector in tf_idf]


class ThemeVectorizer(TFIDFVectorizer):
    def get_vectors(self):
        tf_idf_vectors = super().get_vectors()
        tf_idf_array = np.array(tf_idf_vectors)
        U, s, Vt = np.linalg.svd(tf_idf_array.transpose())

        return tf_idf_array.dot(U)
