from src.utils import ThemeAnalyzer


class DialogManager:
    def __init__(self, answers_mapping: dict, theme_analyzer: ThemeAnalyzer):
        self.answers_mapping = answers_mapping
        self.theme_analyzer = theme_analyzer

    def init_dialog(self):
        print('Здравствуйте, что вас интересует?')

        query = input()
        theme = self.theme_analyzer.get_theme(query)
        answer = 'Извините, не получается распознать ваш запрос'

        if theme is not None:
            answer = self.answers_mapping.get(theme, answer)

        print(answer)
