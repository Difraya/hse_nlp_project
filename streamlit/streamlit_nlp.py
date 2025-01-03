import streamlit as st
import requests
import plotly.express as px
import re
import json
from collections import Counter
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')

st.set_page_config(page_title="Авторство текстов", layout="wide")
pages = ["Пользовательская часть", "Информация про модели и данные", "Обучи свою модель"]
choice = st.sidebar.selectbox("Навигация", pages)
API_URL = "http://127.0.0.1:8000"

def get_top_words(text, n=10):
    words = re.findall(r"\w+", text.lower())
    c = Counter(words).most_common(n)
    return c

def get_ngrams(text, n):
    tokens = word_tokenize(text.lower())
    n_grams = list(ngrams(tokens, n))
    return n_grams

def most_common_ngrams(ngrams_list, top_n=3):
    counter = Counter(ngrams_list)
    return counter.most_common(top_n)

def del_stopwords(text):
    words = word_tokenize(text)
    res = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word.isalpha()]
    return ' '.join(res)

def get_pos(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
    elif isinstance(text, list):
        tokens = text
    else:
        raise TypeError("Ожидалась строка или список слов!")
    pos = [i[1] for i in pos_tag(tokens, lang='eng')]
    pos_count = Counter(pos)
    return pos_count

def punctuation(text):
    return len(re.findall(r'[^\w\s]', text))

if choice == "Пользовательская часть":
    st.title("Пользовательская часть")
    type_inf = st.selectbox("Выберите по каким данным делать анализ и предсказания:", ["Ввести текст", "Загрузить файл"])
    if type_inf == "Ввести текст":
        txt = st.text_area("Введите текст:", "")
        if txt is None:
            st.warning("Введите текст для анализа и обработки")
            st.stop()
        else:
            st.subheader("Анализ текста и обработка текста")
            st.write(f"Общее количество слов: {len(txt.split())}")
            st.write(f"Количество уникальных слов: {len(set(txt.split()))}")
            st.write(f"Количество знаков препинания: {punctuation(txt)}")
            st.warning("Перед построением графиков рекомендуется удалить стоп-слова!")
            if st.checkbox("Удалить стоп-слова"):
                text = del_stopwords(txt)
            else:
                text = txt
            if st.button("Показать распределение частей речи"):
                if text:
                    pos = get_pos(text)
                    df_pos = pd.DataFrame(sorted(pos.items(), key=lambda x: x[1], reverse=True), columns=['part_of_speech', 'count'])
                    fig = px.bar(df_pos, x='part_of_speech', y='count', color='part_of_speech', title='Распределение частей речи')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig)
                else:
                    st.warning("Введите текст для анализа!")
            if text is not None:
                ngram_type = st.selectbox("Выберите тип n-грамм:", ["Униграммы", "Биграммы", "Триграммы"])
                if ngram_type == "Униграммы":
                    n = 1
                elif ngram_type == "Биграммы":
                    n = 2
                else:
                    n = 3
                ngrams_list = get_ngrams(text, n)
                top_ngrams = most_common_ngrams(ngrams_list)
                st.subheader(f"Самые популярные {ngram_type.lower()}:")
                if top_ngrams:
                    labels = [' '.join(gram) for gram, count in top_ngrams]
                    counts = [count for gram, count in top_ngrams]
                    fig = px.bar(x=labels, y=counts, color=labels, labels={'x': 'N-граммы', 'y': 'Частота'}, title=f'Топ-3 {ngram_type.lower()}')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig)
                else:
                    st.write("Недостаточно данных для построения графика.")
                w = get_top_words(txt)
                x = [i[0] for i in w]
                y = [i[1] for i in w]
                df = pd.DataFrame({"Слово": x, "Частота": y})
                fig = px.bar(df, x="Слово", y="Частота", color="Слово", title="Топ слов")
                st.plotly_chart(fig, use_container_width=True)
            if st.button("Предсказать к каким авторам мой текст ближе"):
                if 'selected_model' not in st.session_state or not st.session_state['selected_model']:
                    st.warning("Сначала выберите модель для предсказания")
                    st.warning("Для выбора перейдите на страницу: 'Информация про модели и данные'")
                    if st.button("Перейти к выбору модели"):
                        st.experimental_set_query_params(page="Информация про модели и данные")
                    st.stop()
                st.write(f"Для предсказаний вы используете модель: **{st.session_state['selected_model']}**")
                if txt:
                    r = requests.post(f"{API_URL}/PredictItem", json={"text": txt})
                    if r.status_code == 200:
                        res_json = r.json()
                        st.success("По стилю написания наиболее близок вам:")
                        st.write(res_json['author'])
                    res = requests.post(f"{API_URL}/PredictItemProba", json={"text": txt})
                    if res.status_code == 200:
                        pred = res.json()
                        pred_top3 = dict(list(pred.items())[:3])
                        st.success("Самые близкие вам авторы:")
                        for author, proba in pred_top3.items():
                            st.write(f"{author}: {proba:.4f}")
                        authors = list(pred_top3.keys())
                        values = list(pred_top3.values())
                        fig = px.pie(names=authors, values=values, title="Распределение вероятностей внутри топ-3 авторов")
                        st.plotly_chart(fig)
                    elif res.status_code == 400:
                        st.error("Эта модель не умеет предсказывать вероятности, выберите другую модель.")
                    else:
                        st.error("Ошибка при вычислении вероятностей.")
                else:
                    st.warning("Введите текст для предсказания")
    if type_inf == "Загрузить файл":
        upfile = st.file_uploader("Загрузите файл в формате txt для мультиинференса", type=["txt"])
        if upfile is not None:
            try:
                data = upfile.getvalue().decode('utf-8')
                st.subheader("Анализ и обработка текста")
                st.write(f"Общее количество слов: {len(data.split())}")
                st.write(f"Количество уникальных слов: {len(set(data.split()))}")
                st.write(f"Количество знаков препинания: {punctuation(data)}")
                if st.checkbox("Удалить стоп-слова"):
                    text = del_stopwords(data)
                else:
                    text = data
                if text is not None:
                    ngram_type = st.selectbox("Выберите тип n-грамм:", ["Униграммы", "Биграммы", "Триграммы"])
                    if ngram_type == "Униграммы":
                        n = 1
                    elif ngram_type == "Биграммы":
                        n = 2
                    else:
                        n = 3
                    ngrams_list = get_ngrams(text, n)
                    top_ngrams = most_common_ngrams(ngrams_list)
                    st.subheader(f"Самые популярные {ngram_type.lower()}:")
                    if top_ngrams:
                        labels = [' '.join(gram) for gram, count in top_ngrams]
                        counts = [count for gram, count in top_ngrams]
                        fig = px.bar(x=labels, y=counts, color=labels, labels={'x': 'N-граммы', 'y': 'Частота'}, title=f'Топ-3 {ngram_type.lower()}')
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig)
                    else:
                        st.write("Недостаточно данных для построения графика.")
                    w = get_top_words(text)
                    x = [i[0] for i in w]
                    y = [i[1] for i in w]
                    df = pd.DataFrame({"Слово": x, "Частота": y})
                    fig = px.bar(df, x="Слово", y="Частота", color="Слово", title="Топ слов")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка при загрузке файла: {str(e)}")
        if st.button("Сделать предсказания по файлу"):
            if 'selected_model' not in st.session_state or not st.session_state['selected_model']:
                st.warning("Сначала выберите модель для предсказания")
                st.warning("Для выбора перейдите на страницу: 'Информация про модели и данные'")
                if st.button("Перейти к выбору модели"):
                    st.experimental_set_query_params(page="Информация про модели и данные")
                st.stop()
            st.write(f"Для предсказаний вы используете модель: **{st.session_state['selected_model']}**")
            upfile.seek(0)
            files = {'request': upfile}
            response = requests.post(f"{API_URL}/PredictItemFile", files=files)
            if response.status_code == 200:
                prediction = response.json()
                st.success(f"Результат предсказания: {prediction['author']}")
            else:
                st.error(f"Ошибка при предсказании: {response.status_code}")
            res = requests.post(f"{API_URL}/PredictItemProbaFile", files=files)
            if res.status_code == 200:
                prediction = res.json()
                prediction_top3 = dict(list(prediction.items())[:3])
                st.success("Самые близкие вам авторы:")
                for author, proba in prediction_top3.items():
                    st.write(f"{author}: {proba:.4f}")
                authors = list(prediction_top3.keys())
                values = list(prediction_top3.values())
                fig = px.pie(names=authors, values=values, title="Распределение вероятностей внутри топ-3 авторов")
                st.plotly_chart(fig)
            elif res.status_code == 400:
                st.error("Эта модель не умеет предсказывать вероятности, выберите другую модель.")
            else:
                st.error(f"Ошибка при предсказании: {res.status_code}")

elif choice == "Информация про модели и данные":
    st.title("Информация про модели и данные")
    st.write("Список доступных моделей")
    ml = requests.get(f"{API_URL}/ModelsList")
    if ml.status_code == 200:
        models = ml.json()
        for model in models:
            st.write(f"**{model['name']}**: {model['description']}")
    else:
        st.error("Нет доступных моделей.")
    st.write("Установить активную модель")
    response = requests.get(f"{API_URL}/ModelsList")
    if response.status_code == 200:
        models = response.json()
        model_names = [model['name'] for model in models]
        selected_model = st.selectbox("Выберите модель", model_names)
        if st.button(f"Установить модель {selected_model}"):
            res = requests.post(f"{API_URL}/setModel", params={'mod_id': selected_model})
            if res.status_code == 200:
                st.session_state['selected_model'] = selected_model
                st.success(f"Активная модель установлена: {selected_model}")
            elif res.status_code == 400:
                st.error(f'Модель {selected_model} не существует!')
            else:
                st.error(f"Не удалось установить модель. Ошибка: {res.status_code}")
    else:
        st.error("Нет доступных моделей.")

elif choice == "Обучи свою модель":
    st.title("Обучить свою модель")
    if 'selected_model' not in st.session_state or not st.session_state['selected_model']:
        st.warning("Сначала выберите модель для обучения")
        st.warning("Для выбора перейдите на страницу: 'Информация про модели и данные'")
        if st.button("Перейти к выбору модели"):
            st.query_params(page="Информация про модели и данные")
        st.stop()
    st.write(f"Для обучения вы используете модель: **{st.session_state['selected_model']}**")
    if st.session_state['selected_model'] == 'model1':
        st.warning("На обучение model1 уходит слишком много времени. Активируйте другую модель из списка.")
        if st.button("Перейти к выбору модели"):
            st.experimental_set_query_params(page="Информация про модели и данные")
        st.stop()
    train_file = st.file_uploader("Загрузите тренировочный датасет (Parquet)", type=['pq'])
    test_file = st.file_uploader("Загрузите тестовый датасет (Parquet)", type=['pq'])
    st.header("Настройка гиперпараметров")
    hyperparameters = st.text_area("Гиперпараметры (в формате JSON)", '{}')
    if st.button("Обучить модель"):
        if not train_file or not test_file:
            st.error("Пожалуйста, загрузите оба файла: обучающий и тестовый.")
        else:
            try:
                with st.spinner("Обучение модели..."):
                    files = {
                        "train_file": train_file,
                        "test_file": test_file,
                    }
                    data = {"request": hyperparameters}
                    response = requests.post(f"{API_URL}/train_model", data=data, files=files)
                if response.status_code == 200:
                    train_result = response.json()
                    st.success("Модель успешно обучена!")
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
                else:
                    st.error(f"Ошибка при обучении модели: {response.status_code}")
                    if response.status_code != 404:
                        st.error(response.json().get("detail", "Неизвестная ошибка"))
            except Exception as e:
                st.error(f"Произошла ошибка: {e}")
    st.title("Частичное дообучение (partial_fit)")
    train_file_partial = st.file_uploader("Загрузите датасет для дообучения (Parquet)", type=['pq'], key="partial_fit_file")
    if st.button("Запустить обучение модели SVC"):
        if train_file_partial is not None:
            try:
                train_file_partial.seek(0)
                files = {'request_file': train_file_partial}
                params = {}
                response = requests.post(f"{API_URL}/partial_fit", files=files, data=params)
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
