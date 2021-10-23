from src.utils import ThemeAnalyzer
from src.model.data import corpus, answers_mapping


class DialogManager:
    def __init__(self):
        self.theme_analyzer = ThemeAnalyzer(corpus)

    def init_dialog(self):
        print('Здравствуйте, что вас интересует?')

        query = input()
        theme = self.theme_analyzer.get_theme(query)
        answer = 'Извините, не получается распознать ваш запрос'

        if theme is not None:
            answer = answers_mapping.get(theme, answer)

        print(answer)
