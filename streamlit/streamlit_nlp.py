import streamlit as st
import requests
import plotly.express as px
import re
import os
import io
import json
from collections import Counter
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

st.set_page_config(page_title="Авторство текстов", layout="wide")
pages = ["Предсказание автора", "Информация про модели и данные", "Обучи свою модель"]
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
    ml_response = requests.get(f"{API_URL}/ModelsList")
    if ml_response.status_code == 200:
        models = ml_response.json()
        model_names = [model['name'] for model in models]
        selected_model = model_names[0]
        response = requests.post(f"{API_URL}/setModel", params={'mod_id': selected_model})
        st.session_state['selected_model'] = selected_model
else:
    # Устанавливает модель в API, если она существует в session_state
    response = requests.post(f"{API_URL}/setModel", params={'mod_id': st.session_state['selected_model']})

def get_top_words(text, n=10):
    words = re.findall(r"\w+", text.lower())
    return Counter(words).most_common(n)


def get_ngrams(text, n):
    tokens = word_tokenize(text.lower())
    return list(ngrams(tokens, n))


def most_common_ngrams(ngrams_list, top_n=3):
    return Counter(ngrams_list).most_common(top_n)


def del_stopwords(text):
    words = word_tokenize(text)
    return ' '.join(word.lower() for word in words if word.lower() not in stopwords.words('english') and word.isalpha())


def get_pos(text):
    tokens = word_tokenize(text) if isinstance(text, str) else text
    pos_tags = pos_tag(tokens, lang='eng')
    return Counter([pos for _, pos in pos_tags])


def punctuation(text):
    return len(re.findall(r'[^\w\s]', text))


def analyze_and_display_text(text):
    st.subheader("Анализ текста и обработка текста")
    st.write(f"Общее количество слов: {len(text.split())}")
    st.write(f"Количество уникальных слов: {len(set(text.split()))}")
    st.write(f"Количество знаков препинания: {punctuation(text)}")
    st.warning("Перед построением графиков рекомендуется удалить стоп-слова!")
    return del_stopwords(text) if st.checkbox("Удалить стоп-слова") else text


def plot_pos_distribution(text):
    if st.button("Показать распределение частей речи"):
        pos = get_pos(text)
        df_pos = pd.DataFrame(pos.items(), columns=['part_of_speech', 'count'])
        df_pos.sort_values(by='count', ascending=False, inplace=True)
        fig = px.bar(df_pos, x='part_of_speech', y='count', color='part_of_speech', title='Распределение частей речи')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)


def plot_ngrams_distribution(text, ngram_type):
    n = {'Униграммы': 1, 'Биграммы': 2, 'Триграммы': 3}[ngram_type]
    ngrams_list = get_ngrams(text, n)
    top_ngrams = most_common_ngrams(ngrams_list)
    labels = [' '.join(gram) for gram, _ in top_ngrams]
    counts = [count for _, count in top_ngrams]
    fig = px.bar(x=labels, y=counts, color=labels, labels={'x': 'N-граммы', 'y': 'Частота'}, title=f'Топ-3 {ngram_type.lower()}')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)


def plot_top_words(text):
    words = get_top_words(text)
    df_words = pd.DataFrame(words, columns=["Слово", "Частота"])
    fig = px.bar(df_words, x="Слово", y="Частота", color="Слово", title="Топ слов")
    st.plotly_chart(fig, use_container_width=True)


def get_writer_name_ru(author_name):
    return writers_dict.get(author_name, "Author not found")


def get_author_image(author_name):
    # Поиск изображения автора по имени
    image_file = os.path.join(IMAGES_PATH, f'{author_name}.jpg')
    if os.path.exists(image_file):
        return image_file
    return None


def get_bio_by_author_name(author_name):
    # Поиск биографии автора по имени
    bio = authors_bio[authors_bio['name'] == author_name]['bio']
    return bio.iloc[0] if not bio.empty else "Биография не найдена."


def predict_author_by_text(text):
    # st.write(f"Для предсказаний вы используете модель: {st.session_state['selected_model']}")
    response = requests.post(f"{API_URL}/PredictItem", json={"text": text})
    if response.status_code == 200:
        predicted_author = response.json()['author']
        st.success("По стилю написания наиболее близок вам:")
        author_name_ru = get_writer_name_ru(predicted_author)
        st.write(author_name_ru)
        
        # Вывод изображения автора
        image_path = get_author_image(predicted_author)
        if image_path:
            st.image(image_path, caption=f'Фото {author_name_ru}')
        
        # Вывод биографии автора
        with st.expander("Биография автора на английском языке:", expanded=False):
            author_bio = get_bio_by_author_name(predicted_author)
            st.write(author_bio)
    else:
        st.error("Ошибка при предсказании автора.")

    response_proba = requests.post(f"{API_URL}/PredictItemProba", json={"text": text})
    if response_proba.status_code == 200:
        probabilities = response_proba.json()
        plot_author_probabilities(probabilities)
    elif response_proba.status_code == 400:
        st.error("Эта модель не умеет предсказывать вероятности, выберите другую модель.")
    else:
        st.error("Ошибка при вычислении вероятностей.")


def plot_author_probabilities(probabilities):
    top3 = dict(list(probabilities.items())[:3])
    st.success("Самые близкие вам авторы:")
    for author, proba in top3.items():
        st.write(f"{get_writer_name_ru(author)}: с вероятностью {proba * 100:.2f}%")
    authors, values = zip(*top3.items())
    authors = [get_writer_name_ru(x) for x in authors]
    fig = px.pie(names=authors, values=values, title="Распределение вероятностей внутри топ-3 авторов")
    st.plotly_chart(fig)


def perform_text_analysis(text):
    if not text:
        st.warning("Введите текст на английском языке для анализа и обработки")
        st.stop()

    # Сохраним текст в session_state
    st.session_state['input_text'] = text
    with st.expander("Выполнить анализ текста", expanded=False):
        text = analyze_and_display_text(text)
    
        plot_pos_distribution(text)
        ngram_type = st.selectbox("Выберите тип n-грамм:", ["Униграммы", "Биграммы", "Триграммы"])
        plot_ngrams_distribution(text, ngram_type)
        plot_top_words(text)

    if 'selected_model' in st.session_state and st.session_state['selected_model']:
        st.markdown(f"Активирована модель {st.session_state['selected_model']}")

    if st.button("Предсказать к каким авторам мой текст ближе"):
        if 'selected_model' not in st.session_state or not st.session_state['selected_model']:
            st.warning("Сначала выберите модель для предсказания")
            st.experimental_set_query_params(page="Информация про модели и данные")
            st.stop()
        predict_author_by_text(text)


def handle_file_upload():
    uploaded_file = st.file_uploader(
      "Загрузите файл с текстом на английском языке в формате txt для предсказания",
      type=["txt"])
    if uploaded_file is not None:
        # Сохраним содержимое файла в session_state
        file_content = uploaded_file.getvalue().decode('utf-8')
        st.session_state['uploaded_file'] = file_content
        try:
            file_json = json.dumps({"text": file_content})
            perform_text_analysis(file_json)
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {str(e)}")


def predict_author_by_file(file_like_object):
    if st.button("Сделать предсказания по файлу"):
        if 'selected_model' not in st.session_state or not st.session_state['selected_model']:
            st.warning("Сначала выберите модель для предсказания")
            st.experimental_set_query_params(page="Информация про модели и данные")
            st.stop()
        file_like_object.seek(0)
        files = {'request': file_like_object}
        response = requests.post(f"{API_URL}/PredictItemFile", files=files)
        if response.status_code == 200:
            prediction = response.json()['author']
            st.success(f"Результат предсказания: {get_writer_name_ru(prediction)}")
        else:
            st.error(f"Ошибка при предсказании: {response.status_code}")

        response_prob_file = requests.post(f"{API_URL}/PredictItemProbaFile", files=files)
        if response_prob_file.status_code == 200:
            probabilities = response_prob_file.json()
            plot_author_probabilities(probabilities)
        elif response_prob_file.status_code == 400:
            st.error("Эта модель не умеет предсказывать вероятности, выберите другую модель.")
        else:
            st.error(f"Ошибка при предсказании: {response_prob_file.status_code}")


def handle_model_selection():
    st.title("Информация про модели и данные")
    st.write('''Представленные в приложении модели были обучены на датасете \
с английскими текстами 100 мировых классиков. \
Для выбора авторов за основу был взят \
[рейтинг](https://www.imdb.com/list/ls005774742/).
Данные были получены при помощи парсинга текстов книг с сайтов \
gutenberg.org и loyalbooks.com.''')
    with st.expander("Список доступных моделей", expanded=False):
        ml_response = requests.get(f"{API_URL}/ModelsList")
    
        if ml_response.status_code == 200:
            models = ml_response.json()
            for model in models:
                st.write(f"**{model['name']}**: {model['description']}")
            model_names = [model['name'] for model in models]
            
            # Берем выбранную модель из session_state или первая доступная как дефолт
            selected_model = st.session_state.get('selected_model', model_names[0])

            # Отображаем выбор модели
            selected_model = st.selectbox("Выберите модель", model_names, index=model_names.index(selected_model))

            if st.button(f"Установить модель {selected_model}"):
                response = requests.post(f"{API_URL}/setModel", params={'mod_id': selected_model})
                if response.status_code == 200:
                    st.session_state['selected_model'] = selected_model
                    st.success(f"Активная модель установлена: {selected_model}")
                elif response.status_code == 400:
                    st.error(f'Модель {selected_model} не существует!')
                else:
                    st.error(f"Не удалось установить модель. Ошибка: {response.status_code}")
        else:
            st.error("Нет доступных моделей.")
    st.image('books.jpg', use_container_width=True)


def display_training_comparison_graph():
    """Fetch available experiments and plot their learning curves for comparison."""
    # Fetch experiments from the FastAPI endpoint
    experiments_response = requests.get(f"{API_URL}/experiments")

    if experiments_response.status_code == 200:
        experiments = experiments_response.json()
        experiment_options = {exp['id']: exp['name'] for exp in experiments}

        selected_experiments = st.multiselect(
            "Выберите эксперименты для сравнения", 
            options=list(experiment_options.keys()), 
            format_func=lambda x: experiment_options[x]
        )

        if selected_experiments:
            all_curves_data = {}
            for exp_id in selected_experiments:
                exp_response = requests.get(f"{API_URL}/experiments", params={"exp_id": exp_id})

                if exp_response.status_code == 200:
                    curve_data = exp_response.json()
                    all_curves_data[experiment_options[exp_id]] = curve_data['train_scores_mean']
                    all_curves_data[experiment_options[exp_id] + ' тест'] = curve_data['test_scores_mean']

            if all_curves_data:
                st.line_chart(all_curves_data)
            else:
                st.error("Ошибка при получении данных кривых обучения.")
    else:
        st.error("Не удалось загрузить эксперименты.")


def train_model():
    if 'selected_model' not in st.session_state or not st.session_state['selected_model']:
        st.warning("Сначала выберите модель для обучения")
        st.experimental_set_query_params(page="Информация про модели и данные")
        st.stop()

    train_file = st.file_uploader("Загрузите тренировочный датасет (Parquet)", type=['pq'])
    test_file = st.file_uploader("Загрузите тестовый датасет (Parquet)", type=['pq'])

    st.header("Настройка гиперпараметров")
    hyperparameters = st.text_area("Гиперпараметры (в формате JSON)", '{}')

    if st.button("Обучить модель"):
        if not train_file or not test_file:
            st.error("Пожалуйста, загрузите оба файла: обучающий и тестовый.")
            return

        try:
            with st.spinner("Обучение модели..."):
                files = {"train_file": train_file, "test_file": test_file}
                data = {"request": hyperparameters}
                response = requests.post(f"{API_URL}/train_model", data=data, files=files)

            if response.status_code == 200:
                train_result = response.json()
                st.success("Модель успешно обучена!")
                display_train_results(train_result)
            else:
                st.error(f"Ошибка при обучении модели: {response.status_code}")
                if response.status_code != 404:
                    st.error(response.json().get("detail", "Неизвестная ошибка"))
        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")


def display_train_results(train_result):
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
    train_file_partial = st.file_uploader("Загрузите датасет для дообучения (Parquet)", type=['pq'], key="partial_fit_file")
    if st.button("Запустить дообучение модели SGDClassifier"):
        if train_file_partial:
            try:
                train_file_partial.seek(0)
                files = {'request_file': train_file_partial}
                response = requests.post(f"{API_URL}/partial_fit", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(result["message"])
                elif response.status_code == 400:
                    st.error(response.json()['detail'])
                else:
                    st.error(f"Ошибка при дообучении модели: {response.status_code}")
            except Exception as e:
                st.error(f"Ошибка при обработке данных: {str(e)}")
        else:
            st.warning("Загрузите файл с новыми данными для дообучения.")


if choice == "Предсказание автора":
    st.title("Предсказание автора текста")
    st.write('''Приложение умеет предсказывать вероятности авторства для 100 классиков \
мировой литературы по полученным текстам на англйском языке''')
    st.image('library_hamilton.jpg', use_container_width=True)
    data_input_type = st.selectbox("Выберите по каким данным делать анализ и предсказания:",
                                   ["Ввести текст", "Загрузить файл"])
    if data_input_type == "Ввести текст":
        text_input = st.text_area("Введите текст:", st.session_state['input_text'])
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
