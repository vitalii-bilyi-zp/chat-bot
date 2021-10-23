from src.utils import Tokenizer, ThemeVectorizer, ThemeAnalyzer, DialogManager
from src.model.data import punctuation_symbols, corpus, answers_mapping


if __name__ == '__main__':
    tokenizer = Tokenizer(punctuation_symbols)
    vectorizer = ThemeVectorizer()
    theme_analyzer = ThemeAnalyzer(corpus, tokenizer, vectorizer)
    dialog_manager = DialogManager(answers_mapping, theme_analyzer)
    dialog_manager.init_dialog()
