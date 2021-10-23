import math
from src.utils import Tokenizer, Vectorizer


class ThemeAnalyzer:
    def __init__(self, corpus: list, tokenizer: Tokenizer, vectorizer: Vectorizer):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer

    def get_theme(self, query: str):
        new_corpus = [{'text': query}, *self.corpus]
        vectors = self.get_vectors(new_corpus)

        result_index = None
        prev_sim_coefficient = 0
        min_sim_coefficient = 0.5

        vec1 = vectors[0][0:3]
        for index, vector in enumerate(vectors[1:]):
            vec2 = vector[0:3]
            sim_coefficient = ThemeAnalyzer.cosine_sim(vec1, vec2)

            if sim_coefficient > prev_sim_coefficient and sim_coefficient > min_sim_coefficient:
                result_index = index
                prev_sim_coefficient = sim_coefficient

        if result_index is not None:
            return self.corpus[result_index]['theme']

    def get_vectors(self, corpus: list):
        all_tokens, lexicon = self.tokenizer.tokenize_corpus(corpus)
        return self.vectorizer.get_vectors(all_tokens, lexicon)

    @staticmethod
    def cosine_sim(vec1: list, vec2: list):
        dot_prod = 0
        for i, v in enumerate(vec1):
            dot_prod += v * vec2[i]

        mag_1 = math.sqrt(sum([x ** 2 for x in vec1]))
        mag_2 = math.sqrt(sum([x ** 2 for x in vec2]))

        if mag_1 == 0 or mag_2 == 0:
            return 0

        return dot_prod / (mag_1 * mag_2)
