from src.model.types import Theme


corpus = [
    {'text': 'Хочу записаться к врачу', 'theme': Theme.APPOINTMENT.value},
    {'text': 'Хочу записаться к доктору', 'theme': Theme.APPOINTMENT.value},
    {'text': 'Планирую записаться на прием', 'theme': Theme.APPOINTMENT.value},
    {'text': 'Можно ли попасть на прием к врачу', 'theme': Theme.APPOINTMENT.value},
    {'text': 'Хотел бы попасть на прием в вашей клинике', 'theme': Theme.APPOINTMENT.value},
    {'text': 'Интересует стоимость ваших услуг', 'theme': Theme.SERVICES_COST.value},
    {'text': 'Сколько стоят услуги в вашей клинике?', 'theme': Theme.SERVICES_COST.value},
    {'text': 'Интересует стоимость услуг в вашей клинике', 'theme': Theme.SERVICES_COST.value},
]

answers_mapping = {
    Theme.APPOINTMENT.value: 'Ответ на запрос о записи на прием',
    Theme.SERVICES_COST.value: 'Ответ на запрос стоимости услуг'
}
