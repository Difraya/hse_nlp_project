import streamlit as st
import re
import os
import json
from collections import Counter
import logging
from logging.handlers import RotatingFileHandler
import requests
import plotly.express as px
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
from writers import writers_dict
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')

# Создаем необходимую директорию для логов
os.makedirs("logs", exist_ok=True)

# Настройка логирования
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)

# Настройка обработчика файла с ротацией логов
handler = RotatingFileHandler("logs/app.log", encoding='utf-8',
                              maxBytes=50 * 1024 * 1024, backupCount=5)
handler.setLevel(logging.DEBUG)

# Формат логирования
formatter = logging.Formatter('%(asctime)s - %(name)s - \
                              %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Добавление обработчика в логгер
logger.addHandler(handler)

st.set_page_config(page_title="Авторство текстов", layout="wide")
pages = ["Предсказание автора", "Информация про модели и данные",
         "Обучи свою модель"]
choice = st.sidebar.selectbox("Навигация", pages)

API_URL = 'http://127.0.0.1:8000'
# Для сборки докер-образа нужно закомментировать строку выше
# и раскомментировать строку ниже, вместо неё
# API_URL = 'http://fastapi:8000'

# Путь к папке с фотографиями авторов
IMAGES_PATH = 'images'

# Загрузим датафрейм с биографиями авторов
authors_bio = pd.read_csv('authors_bio.csv')

# Блок инициации состояния сессии для сохранения ввода
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

# Активируем первую модель по умолчанию
if 'selected_model' not in st.session_state:
    ml_response = requests.get(f"{API_URL}/ModelsList", timeout=120)
    if ml_response.status_code == 200:
        models = ml_response.json()
        model_names = [model['name'] for model in models]
        selected_model = model_names[0]
        response = requests.post(f"{API_URL}/setModel",
                                 params={'mod_id': selected_model},
                                 timeout=120)
        st.session_state['selected_model'] = selected_model
        logger.info("Successfully set default model: %s",
                    selected_model)
else:
    # Устанавливает модель в API, если она существует в session_state
    response = requests.post(
        f"{API_URL}/setModel",
        params={'mod_id': st.session_state['selected_model']},
        timeout=120)
    logger.info("Re-confirmed previously selected model: %s",
                st.session_state['selected_model'])


def get_top_words(text, n=10):
    """
    Возвращает 'n' наиболее часто встречающихся слов из входного текста.
    Параметры:
    text (str): Входной текст, из которого извлекаются слова.
    n (int): Количество наиболее часто встречающихся слов для возврата.
             По умолчанию 10.
    Возвращает:
    list: Список кортежей, где каждый кортеж содержит слово и его частоту,
          отсортированный от наиболее к наименее частому.
          Возвращает пустой список, если происходит ошибка.
    Исключения:
    ValueError: Если ввод не является допустимым текстом.
    """
    try:
        words = re.findall(r"\w+", text.lower())
        top_words = Counter(words).most_common(n)
        logger.info("Наиболее часто встречающиеся слова успешно извлечены.")
        return top_words
    except ValueError as e:
        logger.error("Произошла ошибка ValueError: %s", e)
    except KeyError as e:
        logger.error("Произошла ошибка KeyError: %s", e)
    return []


def get_ngrams(text, n):
    """
    Извлекает n-граммы из данного текста.
    :param text: Исходный текст.
    :param n: Длина n-граммы.
    :return: Список n-грамм.
    """
    try:
        tokens = word_tokenize(text.lower())
        ngrams_list = list(ngrams(tokens, n))
        logger.info("N-grams retrieved successfully.")
        return ngrams_list
    except ValueError as e:
        logger.error("Произошла ошибка ValueError: %s", e)
    except KeyError as e:
        logger.error("Произошла ошибка KeyError: %s", e)
    return None


def most_common_ngrams(ngrams_list, top_n=3):
    """
    Находит наиболее часто встречающиеся n-граммы в списке.
    :param ngrams_list: Список n-грамм.
    :param top_n: Количество самых частых n-грамм, которое нужно вернуть.
    :return: Список наиболее частых n-грамм и их частота.
    """
    try:
        common_ngrams = Counter(ngrams_list).most_common(top_n)
        logger.info("Most common n-grams retrieved successfully.")
        return common_ngrams
    except ValueError as e:
        logger.error("Произошла ошибка ValueError: %s", e)
    except KeyError as e:
        logger.error("Произошла ошибка KeyError: %s", e)
    return []


def del_stopwords(text):
    """
    Удаляет стоп-слова из переданного текста.
    :param text: Исходный текст для обработки.
    :return: Текст без стоп-слов.
    """
    try:
        words = word_tokenize(text)
        filtered_text = ' '.join(
            word.lower() for word in words
            if word.lower() not in stopwords.words('english')
            and word.isalpha())
        logger.info("Stopwords removed successfully.")
        return filtered_text
    except ValueError as e:
        logger.error("Error removing stopwords: %s", e)
        return ""


def get_pos(text):
    """
    Получает части речи (POS) из предоставленного текста.
    :param text: Строка текста или уже токенизированный список.
    :return: Counter с количеством каждого POS тега.
    """
    try:
        tokens = word_tokenize(text) if isinstance(text, str)\
            else text
        pos_tags = pos_tag(tokens, lang='eng')
        pos_counter = Counter([pos for _, pos in pos_tags])
        logger.info("POS tags retrieved successfully.")
        return pos_counter
    except ValueError as e:
        logger.error("Error retrieving POS tags: %s", e)
        return None


def punctuation(text):
    """
    Вычисляет количество знаков пунктуации в предоставленной строке текста.
    :param text: Строка текста для анализа.
    :return: Количество знаков пунктуации в тексте.
    """
    try:
        punct_count = len(re.findall(r'[^\w\s]', text))
        logger.info("Punctuation count calculated successfully.")
        return punct_count
    except ValueError as e:
        logger.error("Error calculating punctuation count: %s", e)
        return None


def analyze_and_display_text(text):
    """
    Анализирует и отображает информацию о тексте в приложении Streamlit.
    Функция выполняет анализ текста, включая подсчет общего количества
    слов, количества уникальных слов и знаков препинания.
    Пользователю предоставляется возможность удалить стоп-слова из текста.
    :param text: Строка текста для анализа и отображения.
    :return: Текст, из которого удалены стоп-слова,
             если пользователь выбрал соответствующую опцию;
             исходный текст в противном случае.
    """
    try:
        st.subheader("Анализ текста и обработка текста")
        st.write(f"Общее количество слов: {len(text.split())}")
        st.write(f"Количество уникальных слов: \
                 {len(set(text.split()))}")
        st.write(f"Количество знаков препинания: {punctuation(text)}")
        st.warning("Перед построением графиков рекомендуется \
удалить стоп-слова!")
        logger.info("Text analysis and display completed successfully.")
        return del_stopwords(text) if st.checkbox("Удалить стоп-слова")\
            else text
    except ValueError as e:
        logger.error("Error analyzing and displaying text: %s", e)
        return None


def plot_pos_distribution(text):
    """
    Строит и отображает распределение частей речи текста
    в приложении Streamlit.
    Функция создает график, показывающий количество различных частей речи
    в заданном тексте.
    График строится на основе данных, полученных от функции `get_pos`
    и отображается с помощью Plotly.
    :param text: Строка текста, для которого необходимо построить
                 распределение частей речи.
    :return: None. Функция отображает график в приложении Streamlit.
    """
    try:
        if st.button("Показать распределение частей речи"):
            pos = get_pos(text)
            df_pos = pd.DataFrame(pos.items(),
                                  columns=['part_of_speech', 'count'])
            df_pos.sort_values(by='count', ascending=False,
                               inplace=True)
            fig = px.bar(df_pos, x='part_of_speech', y='count',
                         color='part_of_speech',
                         title='Распределение частей речи')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
            logger.info("POS distribution plot displayed successfully.")
    except ValueError as e:
        logger.error("Error displaying POS distribution plot: %s", e)


def plot_ngrams_distribution(text, ngram_type):
    """
    Строит и отображает распределение n-грамм текста в
    приложении Streamlit.
    Функция создает график, показывающий частоту наиболее часто
    встречающихся n-грамм
    (унниграммы, биграммы или триграммы) в заданном тексте.
    График отображается с помощью Plotly.
    :param text: Строка текста, для которой необходимо построить
                 распределение n-грамм.
    :param ngram_type: Тип n-грамм
                       ('Униграммы', 'Биграммы' или 'Триграммы').
    :return: None. Функция отображает график в приложении Streamlit.
    """
    try:
        n = {'Униграммы': 1, 'Биграммы': 2, 'Триграммы': 3}[ngram_type]
        ngrams_list = get_ngrams(text, n)
        top_ngrams = most_common_ngrams(ngrams_list)
        labels = [' '.join(gram) for gram, _ in top_ngrams]
        counts = [count for _, count in top_ngrams]
        fig = px.bar(x=labels, y=counts, color=labels,
                     labels={'x': 'N-граммы', 'y': 'Частота'},
                     title=f'Топ-3 {ngram_type.lower()}')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)
        logger.info("N-grams distribution plot displayed successfully.")
    except ValueError as e:
        logger.error("Error displaying n-grams distribution plot: %s", e)


def plot_top_words(text):
    """
    Строит и отображает график топ слов в заданном тексте
    в приложении Streamlit.
    Функция создает график, показывающий частоту определенных слов
    в тексте, основываясь на данных, полученных из`get_top_words`.
    График отображается с помощью библиотеки Plotly.
    :param text: Строка текста, для которого необходимо построить
                 график частоты слов.
    :return: None. Функция отображает график в приложении Streamlit.
    """
    try:
        words = get_top_words(text)
        df_words = pd.DataFrame(words, columns=["Слово", "Частота"])
        fig = px.bar(df_words, x="Слово", y="Частота",
                     color="Слово", title="Топ слов")
        st.plotly_chart(fig, use_container_width=True)
        logger.info("Top words plot displayed successfully.")
    except ValueError as e:
        logger.error("Error displaying top words plot: %s", e)


def get_writer_name_ru(author_name):
    """
    Возвращает имя писателя на русском языке, соответствующее
    заданному имени автора.
    Функция ищет имя писателя в словаре `writers_dict`
    на основе переданного имени автора.
    :param author_name: Имя автора, как ключ для поиска в словаре.
    :return: Имя писателя на русском языке
             или сообщение "Author not found",
             если автор не найден. В случае ошибки возвращает None.
    """
    try:
        writer_name = writers_dict.get(author_name, "Author not found")
        logger.info("Writer name retrieved successfully.")
        return writer_name
    except ValueError as e:
        logger.error("Error retrieving writer name: %s", e)
        return None


def get_author_image(author_name):
    """
    Возвращает путь к изображению автора, если оно существует.
    Функция осуществляет поиск изображения автора по имени в
    определенной директории. Если изображение найдено, возвращается
    путь к нему. Если изображение не найдено, возвращается None.
    Параметры:
    author_name (str): Имя автора, для которого требуется найти фото.
    Возвращает:
    str или None: Путь к файлу изображения или None,
                  если изображение не найдено.
    """
    try:
        image_file = os.path.join(IMAGES_PATH, f'{author_name}.jpg')
        if os.path.exists(image_file):
            logger.info("Author image found successfully.")
            return image_file
        logger.info("Author image not found.")
        return None
    except ValueError as e:
        logger.error("Error retrieving author image: %s", e)
        return None


def get_bio_by_author_name(author_name):
    """
    Возвращает биографию автора по его имени.
    Функция ищет биографию автора в заранее подготовленном
    наборе данных.
    Параметры:
    author_name (str): Имя автора, чью биографию необходимо получить.
    Возвращает:
    str: Биография автора или сообщение "Биография не найдена."
    """
    try:
        bio = authors_bio[authors_bio['name'] == author_name]['bio']
        if not bio.empty:
            logger.info("Author bio retrieved successfully.")
            return bio.iloc[0]
        logger.info("Author bio not found.")
        return "Биография не найдена."
    except ValueError as e:
        logger.error("Error retrieving author bio: %s", e)
        return None


def predict_author_by_text(text):
    """
    Предсказывает автора текста и отображает информацию о нём.
    Функция отправляет текст на сервер для прогнозирования автора.
    Если успешное предсказание, отображает имя автора,
    его фото и биографию.
    Также отправляет запрос для получения вероятностей предсказания
    и визуализирует их.
    Параметры:
    text (str): Текст, для которого нужно предсказать автора.
    Возвращает: None
    """
    try:
        # Re-confirm the selected model
        logger.info("Re-confirmed previously selected model: %s",
                    st.session_state['selected_model'])
        resp = requests.post(f"{API_URL}/PredictItem",
                             json={"text": text},
                             timeout=120)
        if resp.status_code == 200:
            predicted_author = resp.json()['author']
            st.success("По стилю написания наиболее близок вам:")
            author_name_ru = get_writer_name_ru(predicted_author)
            st.write(author_name_ru)

            # Display author's image
            image_path = get_author_image(predicted_author)
            if image_path:
                st.image(image_path, caption=f'Фото {author_name_ru}')

            # Display author's biography
            with st.expander("Биография автора на английском языке:",
                             expanded=False):
                author_bio = get_bio_by_author_name(predicted_author)
                st.write(author_bio)
        else:
            st.error("Ошибка при предсказании автора.")

        response_proba = requests.post(f"{API_URL}/PredictItemProba",
                                       json={"text": text},
                                       timeout=120)
        if response_proba.status_code == 200:
            probabilities = response_proba.json()
            plot_author_probabilities(probabilities)
        elif response_proba.status_code == 400:
            st.error("Эта модель не умеет предсказывать вероятности,\
выберите другую модель.")
        else:
            st.error("Ошибка при вычислении вероятностей.")
    except ValueError as e:
        logger.error("Error re-confirming selected model: %s", e)


def plot_author_probabilities(probabilities):
    """
    Отображает топ-3 авторов по вероятностям и визуализирует данные
    в виде круговой диаграммы.
    Функция получает словарь вероятностей авторов, выбирает топ-3
    авторов с наивысшими вероятностями и выводит их имена
    с соответствующими вероятностями. Визуализирует распределение
    вероятностей в виде круговой диаграммы.
    Параметры:
    probabilities (dict): Словарь, где ключи — авторы,
                          а значения — их вероятности.
    Возвращает: None
    """
    top3 = dict(list(probabilities.items())[:3])
    st.success("Самые близкие вам авторы:")
    for author, proba in top3.items():
        st.write(f"{get_writer_name_ru(author)}: с вероятностью \
{proba * 100:.2f}%")
    authors, values = zip(*top3.items())
    authors = [get_writer_name_ru(x) for x in authors]
    fig = px.pie(names=authors, values=values,
                 title="Распределение вероятностей внутри топ-3 авторов")
    st.plotly_chart(fig)


def perform_text_analysis(text):
    """
    Выполняет анализ и обработку текста на английском языке.
    Эта функция проверяет ввод текста, сохраняет его в состоянии сеанса и
    выполняет различные анализы. Включает в себя выполнение анализа текста,
    распределение частей речи, распределение n-грамм и выделение топ слов.
    Включает возможность выбора модели для предсказания наиболее близкого
    автора к введённому тексту.
    Параметры:
    text (str): Текст для анализа.
    Возвращает: None
    """
    if not text:
        st.warning("Введите текст на английском языке для \
анализа и обработки")
        st.stop()

    # Save the text in session_state
    st.session_state['input_text'] = text
    with st.expander("Выполнить анализ текста", expanded=False):
        text = analyze_and_display_text(text)

        plot_pos_distribution(text)
        ngram_type = st.selectbox("Выберите тип n-грамм:",
                                  ["Униграммы", "Биграммы", "Триграммы"])
        plot_ngrams_distribution(text, ngram_type)
        plot_top_words(text)

    if ('selected_model' in st.session_state and
        st.session_state['selected_model']):
        st.markdown(
              f"Активирована модель {st.session_state['selected_model']}")

    if st.button("Предсказать к каким авторам мой текст ближе"):
        if ('selected_model' not in st.session_state or not
            st.session_state['selected_model']):
            st.warning("Сначала выберите модель для предсказания")
            st.experimental_set_query_params(
                page="Информация про модели и данные")
            st.stop()
        predict_author_by_text(text)


def handle_file_upload():
    """
    Обрабатывает загрузку файла и выполняет анализ текста.
    Эта функция позволяет пользователю загрузить текстовый файл
    в формате .txt через пользовательский интерфейс Streamlit.
    Если файл загружен успешно, его содержимое сохраняется в состоянии
    сеанса, а затем передается для выполнения текстового анализа.
    Обрабатываются возможные ошибки, возникающие при попытке прочитать
    или декодировать содержимое файла.
    Возвращает: None
    """
    uploaded_file = st.file_uploader(
        "Загрузите файл с текстом на английском языке в формате \
txt для предсказания",
        type=["txt"])
    if uploaded_file is not None:
        # Сохраним содержимое файла в session_state
        file_content = uploaded_file.getvalue().decode('utf-8')
        st.session_state['uploaded_file'] = file_content
        try:
            file_json = json.dumps({"text": file_content})
            perform_text_analysis(file_json)
        except ValueError as e:
            st.error(f"Ошибка при загрузке файла: {str(e)}")


def predict_author_by_file(file_like_object):
    """
    Выполняет предсказание автора текста по загруженному файлу.
    Эта функция взаимодействует с API для выполнения предсказания
    автора текста на основе содержимого файла,
    загруженного пользователем. Предварительно проверяет, выбрана ли
    модель для предсказания; если нет, выводит предупреждение и
    останавливает выполнение. После успешного предсказания отображает
    результат, а также строит график вероятностей предсказания для
    различных авторов, если модель это поддерживает.
    Параметры:
    file_like_object: Аналог объекта файла,
                      содержащего текст для анализа.
    Возвращает: None
    """
    if st.button("Сделать предсказания по файлу"):
        if ('selected_model' not in st.session_state or not
            st.session_state['selected_model']):
            st.warning("Сначала выберите модель для предсказания")
            st.experimental_set_query_params(
                page="Информация про модели и данные")
            st.stop()
        # Логирование выбранной модели
        logger.info("Re-confirmed previously selected model: %s",
                    st.session_state['selected_model'])

        file_like_object.seek(0)
        files = {'request': file_like_object}
        resp = requests.post(f"{API_URL}/PredictItemFile",
                             files=files,
                             timeout=120)
        if resp.status_code == 200:
            prediction = resp.json()['author']
            st.success(f"Результат предсказания: \
{get_writer_name_ru(prediction)}")
        else:
            st.error(f"Ошибка при предсказании: {resp.status_code}")
        response_prob_file = requests.post(
            f"{API_URL}/PredictItemProbaFile",
            files=files,
            timeout=120)
        if response_prob_file.status_code == 200:
            probabilities = response_prob_file.json()
            plot_author_probabilities(probabilities)
        elif response_prob_file.status_code == 400:
            st.error("Эта модель не умеет предсказывать вероятности, \
выберите другую модель.")
        else:
            st.error(f"Ошибка при предсказании: \
{response_prob_file.status_code}")


def handle_model_selection():
    """
    Отображает интерфейс для выбора и установки модели
    в приложении Streamlit.
    Функция загружает список доступных моделей из API,
    позволяет пользователю выбрать модель из выпадающего списка
    и установить её с помощью POST-запроса.
    Также отображается информация о моделях и изображение книги.
    """
    st.title("Информация про модели и данные")
    st.write('''Представленные в приложении модели были обучены на датасете \
с английскими текстами 100 мировых классиков. \
Для выбора авторов за основу был взят \
[рейтинг](https://www.imdb.com/list/ls005774742/).
Данные были получены при помощи парсинга текстов книг с сайтов \
gutenberg.org и loyalbooks.com.''')
    with st.expander("Список доступных моделей", expanded=False):
        ml_response = requests.get(f"{API_URL}/ModelsList",
                                   timeout=120)
        if ml_response.status_code == 200:
            models = ml_response.json()
            for model in models:
                st.write(f"**{model['name']}**: {model['description']}")
            model_names = [model['name'] for model in models]
            # Берем выбранную модель из session_state
            # или первая доступная как дефолт
            selected_model = st.session_state.get('selected_model',
                                                  model_names[0])
            # Отображаем выбор модели
            selected_model = st.selectbox(
                "Выберите модель",
                model_names,
                index=model_names.index(selected_model))
            if st.button(f"Установить модель {selected_model}"):
                resp = requests.post(f"{API_URL}/setModel",
                                     params={'mod_id': selected_model},
                                     timeout=120)
                if resp.status_code == 200:
                    st.session_state['selected_model'] = selected_model
                    st.success(
                        f"Активная модель установлена: {selected_model}")
                elif resp.status_code == 400:
                    st.error(f'Модель {selected_model} не существует!')
                else:
                    st.error(f"Не удалось установить модель. \
Ошибка: {resp.status_code}")
        else:
            st.error("Нет доступных моделей.")
    st.image('books.jpg', use_container_width=True)


def display_training_comparison_graph():
    """
    Отображает график сравнения кривых обучения для
    выбранных экспериментов. Запрашивает список экспериментов
    и позволяет пользователю выбрать нужные, после чего отображает
    графики трендовых и тестовых средних значений.
    """
    logging.debug("Starting display_training_comparison_graph()")
    experiments_response = requests.get(f"{API_URL}/experiments",
                                        timeout=120)
    logging.debug("Received experiments response with status %s",
                  experiments_response.status_code)
    if experiments_response.status_code == 200:
        experiments = experiments_response.json()
        experiment_options = {exp['id']: exp['name'] for exp in experiments}
        logging.debug("Experiment options: %s", experiment_options)
        selected_experiments = st.multiselect(
            "Выберите эксперименты для сравнения",
            options=list(experiment_options.keys()),
            format_func=lambda x: experiment_options[x]
        )
        logging.debug("Selected experiments: %s", selected_experiments)
        if selected_experiments:
            all_curves_data = {}
            for exp_id in selected_experiments:
                exp_response = requests.get(f"{API_URL}/experiments",
                                            params={"exp_id": exp_id},
                                            timeout=120)
                logging.debug("Received response for exp_id %s with status %s",
                              exp_id, exp_response.status_code)
                if exp_response.status_code == 200:
                    curve_data = exp_response.json()
                    all_curves_data[experiment_options[exp_id]] = \
                        curve_data['train_scores_mean']
                    all_curves_data[experiment_options[exp_id] + ' тест'] = \
                        curve_data['test_scores_mean']
                    logging.debug("Updated curve_data for exp_id %s: %s",
                                  exp_id, curve_data)
            if all_curves_data:
                st.line_chart(all_curves_data)
            else:
                st.error("Ошибка при получении данных кривых обучения.")
                logging.error("No curve data available.")
    else:
        st.error("Не удалось загрузить эксперименты.")
        logging.error("Failed to load experiments with status: %s",
                      experiments_response.status_code)


def train_model():
    """
    Предоставляет интерфейс для обучения модели на загруженных данных.
    Функция позволяет пользователю выбрать модель, загрузить обучающий
    и тестовый наборы данных, указать гиперпараметры в формате JSON
    и запустить процесс обучения модели.
    После успешного обучения отображаются результаты.
    """
    logging.debug("Starting train_model()")
    if ('selected_model' not in st.session_state or not
        st.session_state['selected_model']):
        st.warning("Сначала выберите модель для обучения")
        st.experimental_set_query_params(
            page="Информация про модели и данные")
        st.stop()

    train_file = st.file_uploader(
        "Загрузите тренировочный датасет (Parquet)", type=['pq'])
    test_file = st.file_uploader(
        "Загрузите тестовый датасет (Parquet)", type=['pq'])

    st.header("Настройка гиперпараметров")
    hyperparameters = st.text_area("Гиперпараметры (в формате JSON)", '{}')

    if st.button("Обучить модель"):
        if not train_file or not test_file:
            st.error("Пожалуйста, загрузите оба файла: обучающий и тестовый.")
            logging.error("Training or test file not uploaded.")
            return
        try:
            with st.spinner("Обучение модели..."):
                files = {"train_file": train_file, "test_file": test_file}
                data = {"request": hyperparameters}
                resp = requests.post(f"{API_URL}/train_model",
                                     data=data,
                                     files=files,
                                     timeout=360)
                logging.debug("Training response received with status: %s",
                              resp.status_code)
            if resp.status_code == 200:
                train_result = resp.json()
                st.success("Модель успешно обучена!")
                display_train_results(train_result)
                logging.debug("Training result: %s", train_result)
            else:
                st.error(f"Ошибка при обучении модели: {resp.status_code}")
                logging.error("Error during training with status: %s",
                              resp.status_code)
                if resp.status_code != 404:
                    st.error(resp.json().get("detail",
                             "Неизвестная ошибка"))
        except ValueError as e:
            st.error(f"Произошла ошибка: {str(e)}")
            logging.exception("Exception occurred during model training")


def display_train_results(train_result):
    """
    Отображает результаты обучения модели.
    Параметры:
    train_result (dict): словарь с результатами обучения модели,
                         включая метрики и средние значения оценок.
    """
    st.write(f"**ID модели:** {train_result['mod_id']}")
    st.write(f"**Время выполнения:** {train_result['execution_time']}")
    st.write(f"**Accuracy:** {train_result['accuracy']}")
    st.write(f"**Precision:** {train_result['precision']}")
    st.write(f"**Recall:** {train_result['recall']}")
    st.write(f"**F1 Score:** {train_result['f1']}")
    st.line_chart({
        "Train Scores Mean": train_result["train_scores_mean"],
        "Test Scores Mean": train_result["test_scores_mean"]
    })


def partial_fit_model():
    """
    Выполняет частичное обучение модели SGDClassifier
    на предоставленных данных.
    """
    logging.debug("Starting partial_fit_model()")
    train_file_partial = st.file_uploader(
        "Загрузите датасет для дообучения (Parquet)",
        type=['pq'], key="partial_fit_file")
    if st.button("Запустить дообучение модели SGDClassifier"):
        if train_file_partial:
            try:
                train_file_partial.seek(0)
                files = {'request_file': train_file_partial}
                resp = requests.post(f"{API_URL}/partial_fit",
                                     files=files,
                                     timeout=120)
                logging.debug("Partial fit response received with status: %s",
                              resp.status_code)

                if resp.status_code == 200:
                    result = resp.json()
                    st.success(result["message"])
                    logging.debug("Partial fit success message: %s",
                                  result["message"])
                elif resp.status_code == 400:
                    st.error(resp.json()['detail'])
                    logging.error("Partial fit error detail: %s",
                                  resp.json()['detail'])
                else:
                    st.error(
                      f"Ошибка при дообучении модели: {resp.status_code}")
                    logging.error("Error during partial fit with status: %s",
                                  resp.status_code)
            except ValueError as e:
                st.error(f"Ошибка при обработке данных: {str(e)}")
                logging.exception("Exception during partial fit processing")
        else:
            st.warning("Загрузите файл с новыми данными для дообучения.")


if choice == "Предсказание автора":
    st.title("Предсказание автора текста")
    st.write('''Приложение умеет предсказывать вероятности авторства \
для 100 классиков мировой литературы \
по полученным текстам на англйском языке''')
    st.image('library_hamilton.jpg', use_container_width=True)
    data_input_type = st.selectbox(
        "Выберите по каким данным делать анализ и предсказания:",
        ["Ввести текст", "Загрузить файл"])
    if data_input_type == "Ввести текст":
        text_input = st.text_area("Введите текст:",
                                  st.session_state['input_text'])
        perform_text_analysis(text_input)
    elif data_input_type == "Загрузить файл":
        handle_file_upload()

elif choice == "Информация про модели и данные":
    handle_model_selection()

elif choice == "Обучи свою модель":
    st.title("Обучить свою модель")
    train_model()
    st.header("Сравнить эксперименты")
    display_training_comparison_graph()
    st.write("Дообучить модель SGDClassifier")
    partial_fit_model()
