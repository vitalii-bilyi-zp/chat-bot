import copy
import nltk
import math
import pymorphy2
import numpy as np
from enum import Enum
nltk.download('stopwords')

from nltk.corpus import stopwords
from collections import OrderedDict, Counter


class Theme(Enum):
    APPOINTMENT = 1
    SERVICES_COST = 2


corpus = [
    {'text': 'Хочу записаться к врачу', 'theme': Theme.APPOINTMENT.value},
    {'text': 'Хочу записаться к доктору', 'theme': Theme.APPOINTMENT.value},
    {'text': 'Планирую записаться на прием', 'theme': Theme.APPOINTMENT.value},
    {'text': 'Можно ли попасть на прием к врачу', 'theme': Theme.APPOINTMENT.value},
    {'text': 'Хотел бы попасть на прием в вашей клинике', 'theme': Theme.APPOINTMENT.value},
    {'text': 'Интерисует стоимость ваших услуг', 'theme': Theme.SERVICES_COST.value},
    {'text': 'Сколько стоят услуги в вашей клинике?', 'theme': Theme.SERVICES_COST.value},
    {'text': 'Интерисует стоимость услуг в вашей клинике', 'theme': Theme.SERVICES_COST.value},
]


def tokenize_text(text):
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)

    punctuation_symbols = ['!', '@', '"', '“', '’', '«', '»', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '—',
                           '/', ':', ';', '<', '=', '>', '?', '^', '_', '`', '{', '|', '}', '~', '[', ']']
    ru_stopwords = stopwords.words('russian')
    morph = pymorphy2.MorphAnalyzer()
    return [morph.normal_forms(word)[0]
            for word in tokens
            if (word[0] not in punctuation_symbols and word not in ru_stopwords)]


def tokenize_corpus(corpus):
    all_tokens = []
    lexicon = set()

    for doc in corpus:
        text_tokens = sorted(tokenize_text(doc['text']))
        all_tokens.append(text_tokens)
        lexicon.update(text_tokens)

    return all_tokens, sorted(lexicon)


def get_tf_idf_vectors(all_tokens, lexicon, include_idf=True):
    zero_vector = OrderedDict((token, 0) for token in lexicon)

    tf_idf = []
    idf_mapping = {}
    if include_idf:
        num_docs = len(all_tokens)
        for token in lexicon:
            num_docs_containing_token = 0
            for text_tokens in all_tokens:
                if token in text_tokens:
                    num_docs_containing_token += 1

            if num_docs_containing_token == 0:
                idf_mapping[token] = 1
            else:
                idf_mapping[token] = num_docs / num_docs_containing_token

    for text_tokens in all_tokens:
        vector = copy.copy(zero_vector)
        token_counts = Counter(text_tokens)
        for token, value in token_counts.items():
            tf = value / len(lexicon)

            idf = 1
            if include_idf:
                idf = idf_mapping.get(token, 1)

            vector[token] = tf * idf

        tf_idf.append(vector)

    return [list(vector.values()) for vector in tf_idf]


def get_theme_vectors(tf_idf_vectors):
    tf_idf_array = np.array(tf_idf_vectors)
    U, s, Vt = np.linalg.svd(tf_idf_array.transpose())

    return tf_idf_array.dot(U)


def cosine_sim(vec1, vec2):
    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]

    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))

    return dot_prod / (mag_1 * mag_2)


def get_theme(query):
    new_corpus = [{'text': query}, *corpus]
    tokens, lexicon = tokenize_corpus(new_corpus)
    tf_idf_vectors = get_tf_idf_vectors(tokens, lexicon)
    theme_vectors = get_theme_vectors(tf_idf_vectors)

    result_index = None
    prev_sim_coefficient = 0
    min_sim_coefficient = 0.5

    vec1 = theme_vectors[0][0:3]
    for index, vector in enumerate(theme_vectors[1:]):
        vec2 = vector[0:3]
        sim_coefficient = cosine_sim(vec1, vec2)

        if sim_coefficient > prev_sim_coefficient and sim_coefficient > min_sim_coefficient:
            result_index = index
            prev_sim_coefficient = sim_coefficient

    if result_index is not None:
        return corpus[result_index]['theme']


def start_dialog():
    answers_mapping = {
        Theme.APPOINTMENT.value: 'Ответ на запрос о записи на прием',
        Theme.SERVICES_COST.value: 'Ответ на запрос стоимости услуг'
    }

    print('Здравствуйте, что вас интерисует?')

    query = input()
    theme = get_theme(query)
    answer = 'Извините, не получается распознать ваш запрос'

    if theme is not None:
        answer = answers_mapping.get(theme, answer)

    print(answer)


if __name__ == '__main__':
    start_dialog()
