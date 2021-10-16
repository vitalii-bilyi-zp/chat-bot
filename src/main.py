import copy
import nltk
import math
import pymorphy2
import numpy as np
from abc import ABC, abstractmethod

# download stopwords if they were not installed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from collections import OrderedDict, Counter
from src.model.data import corpus, answers_mapping


class Tokenizer:
    def __init__(self, punctuation_symbols=None):
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.morph = pymorphy2.MorphAnalyzer()
        self.ru_stopwords = stopwords.words('russian')

        if punctuation_symbols is None:
            self.punctuation_symbols = []
        else:
            self.punctuation_symbols = punctuation_symbols

    def tokenize_corpus(self, corpus):
        all_tokens = []
        lexicon = set()

        for doc in corpus:
            text_tokens = sorted(self.tokenize_text(doc['text']))
            all_tokens.append(text_tokens)
            lexicon.update(text_tokens)

        return all_tokens, sorted(lexicon)

    def tokenize_text(self, text):
        tokens = self.tokenizer.tokenize(text)
        return [self.morph.normal_forms(word)[0]
                for word in tokens
                if (word[0] not in self.punctuation_symbols and word not in self.ru_stopwords)]


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


class ThemeAnalyzer:
    def __init__(self):
        self.tokenizer = Tokenizer(['!', '@', '"', '“', '’', '«', '»', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',',
                                    '—', '/', ':', ';', '<', '=', '>', '?', '^', '_', '`', '{', '|', '}', '~', '[', ']'])

    def get_theme(self, query):
        new_corpus = [{'text': query}, *corpus]
        theme_vectors = self.get_theme_vectors(new_corpus)

        result_index = None
        prev_sim_coefficient = 0
        min_sim_coefficient = 0.5

        vec1 = theme_vectors[0][0:3]
        for index, vector in enumerate(theme_vectors[1:]):
            vec2 = vector[0:3]
            sim_coefficient = ThemeAnalyzer.cosine_sim(vec1, vec2)

            if sim_coefficient > prev_sim_coefficient and sim_coefficient > min_sim_coefficient:
                result_index = index
                prev_sim_coefficient = sim_coefficient

        if result_index is not None:
            return corpus[result_index]['theme']

    def get_theme_vectors(self, corpus):
        all_tokens, lexicon = self.tokenizer.tokenize_corpus(corpus)
        vectorizer = ThemeVectorizer(all_tokens, lexicon)

        return vectorizer.get_vectors()

    @staticmethod
    def cosine_sim(vec1, vec2):
        dot_prod = 0
        for i, v in enumerate(vec1):
            dot_prod += v * vec2[i]

        mag_1 = math.sqrt(sum([x ** 2 for x in vec1]))
        mag_2 = math.sqrt(sum([x ** 2 for x in vec2]))

        if mag_1 == 0 or mag_2 == 0:
            return 0

        return dot_prod / (mag_1 * mag_2)


class DialogManager:
    def __init__(self):
        self.theme_analyzer = ThemeAnalyzer()

    def init_dialog(self):
        print('Здравствуйте, что вас интересует?')

        query = input()
        theme = self.theme_analyzer.get_theme(query)
        answer = 'Извините, не получается распознать ваш запрос'

        if theme is not None:
            answer = answers_mapping.get(theme, answer)

        print(answer)


if __name__ == '__main__':
    dialog_manager = DialogManager()
    dialog_manager.init_dialog()
