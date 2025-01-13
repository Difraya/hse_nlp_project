import telebot
import joblib
import os
import pandas as pd
from PIL import Image
import logging
from writers import writers_dict
from pathlib import Path

# Определяем путь к директории и файлу логов
log_directory = 'logs'
log_filename = 'logsfilename.log'
log_filepath = os.path.join(log_directory, log_filename)

# Проверяем, существует ли директория, и создаем ее, если нет
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Настраиваем логирование
logging.basicConfig(filename=log_filepath, level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s - %(message)s')
# Определите путь к файлу модели
current_dir = Path(__file__).resolve().parent
model_path = current_dir.parent/'FastAPI'/'models'/'pipeline.joblib'

# AuthorPredictorBot
API_KEY = ''
IMAGES_PATH = 'images'

bot=telebot.TeleBot(API_KEY)

logging.basicConfig(filename='logs/logsfilename.log', level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s - %(message)s')

def predict_proba(model, text):
    text_series = pd.Series([text])

    return model.predict_proba(text_series)[0]


def get_author_image(author_name):
    try:
        image_file = os.path.join(IMAGES_PATH, f'{author_name}.jpg')
        if os.path.exists(image_file):
            logging.info("Author image found successfully.")
            return image_file
        logging.info("Author image not found.")
        return None
    except ValueError as e:
        logging.error("Error retrieving author image: %s", e)
        return None


def get_writer_name_ru(author_name):
    try:
        writer_name = writers_dict.get(author_name, "Author not found")
        logging.info("Writer name retrieved successfully.")
        return writer_name
    except ValueError as e:
        logging.error("Error retrieving writer name: %s", e)
        return None


@bot.message_handler(commands = ['start'])
def start_message(message):
    bot.send_message(
        message.chat.id,
        'Здравствуйте ' + message.from_user.first_name + ', пожалуйста, введите Ваш текст, чтобы получить предсказания автора')


@bot.message_handler(content_types=['text'])
def send_text(message):
    text = message.text
    model = joblib.load(model_path)
    try:
        probabilities = predict_proba(model, text)
        author_probas = dict(zip(model.classes_, probabilities))
        top_authors = dict(sorted(author_probas.items(),
                       key=lambda item: item[1], reverse=True))
        first_three_pairs = {k: top_authors[k] for k in list(top_authors)[:3]}
        # Переключатель для первого автора
        first_author = True
        for author, prob in first_three_pairs.items():
            if first_author:
                # Загрузка и отправка изображения первого автора
                image = Image.open(get_author_image(author))
                bot.send_photo(message.chat.id, photo=image)
                # Отправка имени автора и вероятности
                bot.send_message(message.chat.id, f"Автором данного текста является:\n{get_writer_name_ru(author)},\nс вероятностью: {prob * 100:.2f}%")
                bot.send_message(message.chat.id, 'Ближайшие по стилю авторы:')
                first_author = False
            else:
                # Отправка имени автора и вероятности для остальных авторов
                bot.send_message(message.chat.id, f"{get_writer_name_ru(author)}: {prob * 100:.2f}%")

    except KeyError:
        logging.error("Проблема с получением данных из ответа.")
        bot.send_message(message.chat.id, 'Не удалось получить авторов, попробуйте ещё раз.')

bot.polling()
