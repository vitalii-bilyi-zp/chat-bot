import nltk
import pymorphy2

# download stopwords if they were not installed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords


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
