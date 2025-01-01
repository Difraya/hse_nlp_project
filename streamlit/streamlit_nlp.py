import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import re
import json
from collections import Counter

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download("punkt")
nltk.download("stopwords")

st.set_page_config(page_title="Авторство текстов", layout="wide")
pages = ["Пользовательская часть", "Информация про модели и данные", "Обучи свою модель"]
choice = st.sidebar.selectbox("Навигация", pages)
API_URL = "http://127.0.0.1:8000"

# ф-ция для получания 10-ти распространных слов
def get_top_words(text, n=10):
    words = re.findall(r"\w+", text.lower())
    c = Counter(words).most_common(n)
    return c

# ф-ция для создания н-грамм
def get_ngrams(text, n):
    tokens = word_tokenize(text.lower())
    n_grams = list(ngrams(tokens, n))
    return n_grams

# ф-ция для отображения распространенных н-грамм
def most_common_ngrams(ngrams_list, top_n=3):
    counter = Counter(ngrams_list)
    return counter.most_common(top_n)

# ф-ция для удаления стоп слов
def del_stopwords(text):
    words = word_tokenize(text)
    res = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word.isalpha()]
    return res

# ф-ция для получения частей речи
def get_pos(text):
    pos = [i[1] for i in pos_tag(text)]
    pos_count = Counter(pos)
    return pos_count

if choice == "Пользовательская часть":
    st.title("Пользовательская часть")
    # обработка фрагмента текста
    type_inf = st.selectbox("Выберите по каким данным делать анализ и предсказания:", ["Ввести текст", "Загрузить файл"])
    if type_inf == "Ввести текст":
      txt = st.text_area("Введите текст:", "")

      if txt is None:
          st.warning("Введите текст для анализа и обработки")
          st.stop()
      st.subheader("Анализ текста и обработка текста")
      if st.button("Удалить стоп-слова"):
        text = del_stopwords(txt)
      else:
        text = txt

      if st.button("Показать распределение частей речи"):
        if text:
            pos = get_pos(text)
            df_pos = pd.DataFrame(sorted(pos.items(), key=lambda x: x[1], reverse=True), columns=['part_of_speech', 'count'])
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=df_pos['part_of_speech'], y=df_pos['count'], ax=ax)
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title('Распределение частей речи')
            st.pyplot(fig)
        else:
            st.warning("Введите текст для анализа!")

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
        fig, ax = plt.subplots()
        ax.bar(labels, counts, color='skyblue')
        ax.set_ylabel("Частота")
        ax.set_title(f"Топ-3 {ngram_type.lower()}")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        st.pyplot(fig)
      else:
        st.write("Недостаточно данных для построения графика.")

      w = get_top_words(txt)
      x = [i[0] for i in w]
      y = [i[1] for i in w]
      fig = px.bar(x=x, y=y, title="Топ слов")
      st.plotly_chart(fig, use_container_width=True)

      if st.button("Предсказать к каким авторам мой текст ближе"):

        if 'selected_model' not in st.session_state or not st.session_state['selected_model']:
          st.warning("Сначала выберите модель для предсказания")
          if st.button("Перейти к выбору модели"):
            st.experimental_set_query_params(page="Информация про модели и данные")
          st.stop()

        st.write(f"Для предсказаний вы используете модель: **{st.session_state['selected_model']}**")

        if txt:
          r = requests.post(f"{API_URL}/PredictItem", json={"text": txt})
          if r.status_code == 200:
            st.success("По стилю написания наиболее близок вам:")
            st.write(r.json())
          res = requests.post(f"{API_URL}/PredictItemProba", json={"text": txt})
          if res.status_code == 200:
            pred = res.json()
            st.success("Самые близкие вам авторы в процентах:")
            for author, proba in pred.items():
              st.write(f"{author}: {proba:.4f}")
        else:
          st.warning("Введите текст для предсказания")

    # обработка датасета
    if type_inf == "Загрузить файл":
      upfile = st.file_uploader("Загрузите файл в формате parquet для мультиинференса", type=["parquet"])

      if upfile is not None:
        try:
          data = pd.read_parquet(upfile)
          st.success("Ваш файл успешно загружен")
          st.write("Первые 10 строк из вашего датасета:")
          st.dataframe(data.head(10))
          st.write(f"Количество строк: {data.shape[0]}, количество столбцов {data.shape[1]}")
          st.write("Описательная статистика:")
          st.write(data.describe(include='object'))
        except Exception as e:
          st.error(f"Ошибка при загрузке файла: {str(e)}")

      st.subheader("Анализ и обработка датасета")
        # для сокращения времени загрузки графиков выберем 5 случайных текстов для визуализации
      df = data.sample(5)

      if st.button("Сделать предсказания по файлу"):

        if 'selected_model' not in st.session_state or not st.session_state['selected_model']:
          st.warning("Сначала выберите модель для предсказания")
          if st.button("Перейти к выбору модели"):
            st.experimental_set_query_params(page="Информация про модели и данные")
          st.stop()

        st.write(f"Для предсказаний вы используете модель: **{st.session_state['selected_model']}**")

        pred_type = st.radio(
            "Выберите тип предсказания:",
            ["Обычное предсказание", "Предсказание с вероятностями"]
        )

        upfile.seek(0)
        files = {'request': upfile}

        if pred_type == "Обычное предсказание":
          response = requests.post(f"{API_URL}/PredictItemFile", files=files)
          if response.status_code == 200:
            prediction = response.json()
            st.write(f"Результат предсказания: {prediction['author']}")
          else:
            st.error(f"Ошибка при предсказании: {response.status_code}")

        if pred_type == "Предсказание с вероятностями":
         response = requests.post(f"{API_URL}/PredictItemProbaFile", files=files)
         if response.status_code == 200:
          prediction = response.json()
          for author, prob in prediction.items():
            st.write(f"{author}: {prob:.4f}")
         else:
          st.error(f"Ошибка при предсказании: {response.status_code}")

# Вывод списка моделей
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
# установка активной модели
    st.write("Установить активную модель")
    response = requests.get(f"{API_URL}/ModelsList")
    if response.status_code == 200:
        models = response.json()
        model_names = [model['name'] for model in models]
        selected_model = st.selectbox("Выберите модель", model_names)

        if st.button(f"Установить модель {selected_model}"):
            headers = {'Content-Type': 'application/json'}
            js = {'model_name': selected_model}
            res = requests.post(f"{API_URL}/setModel", params={"id": selected_model}, json=js, headers=headers)
            if res.status_code == 200:
              st.session_state['selected_model'] = selected_model
              st.success(f"Активная модель установлена: {selected_model}")
            elif res.status_code == 400:
              st.error(f'Модель {selected_model} не существует!')
            else:
              st.error(f"Не удалось установить модель. Ошибка: {res.status_code}")
    else:
        st.error("Нет доступных моделей.")

elif choice == "Обучить свою модель":
    st.title("Обучить свою модель")

    if 'selected_model' not in st.session_state or not st.session_state['selected_model']:
          st.warning("Сначала выберите модель для обучения")
          if st.button("Перейти к выбору модели"):
            st.experimental_set_query_params(page="Информация про модели и данные")
          st.stop()

    st.write(f"Для обучения вы используете модель: **{st.session_state['selected_model']}**")

    if st.session_state['selected_model'] == 'model1':
          st.warning(
              "На обучение model1 уходит слишком много времени. "
              "Активируйте другую модель из списка."
              )
          if st.button("Перейти к выбору модели"):
            st.experimental_set_query_params(page="Информация про модели и данные")
          st.stop()

    train_file = st.file_uploader("Загрузите тренировочный датасет (файл формата Parquet)", type=['parquet'])
    test_file = st.file_uploader("Загрузите тестовый датасет (файл формата Parquet)", type=['parquet'])

    # выбор гиперпараметров
    st.header("Настройка гиперпараметров")
    random_state = st.number_input("random_state", value=42)
    max_iter = st.number_input("max_iter", value=1000, min_value=100, step=100)
    tol = st.number_input("tol", value=1e-4, format="%.1e")

    if st.button("Запуск обучения"):
      if train_file is not None and test_file is not None:
        try:
          train_df = pd.read_parquet(train_file)
          test_df = pd.read_parquet(test_file)
          if train_df.shape[1] != test_df.shape[1]:
            st.error("Размеры тестового и тренировочного датасетов не совпадают")
            st.stop()
          if 'text' not in train_df.columns or 'author' not in train_df.columns:
            st.error("Тренировочный датасет должен содержать колонки 'text' и 'author'")
            st.stop()
          if 'text' not in test_df.columns or 'author' not in test_df.columns:
            st.error("Тестовый датасет должен содержать колонки 'text' и 'author'")
            st.stop()

          hyperparameters = {
              "hyperparameters": {
                  "random_state": random_state,
                  "max_iter": max_iter,
                  "tol": tol
              }
          }
          params = {'request': json.dumps(hyperparameters)}

          train_file.seek(0)
          test_file.seek(0)

          files = {"train_file": train_file, "test_file": test_file}

          response = requests.post(f"{API_URL}/train_model", data=params, files=files)
          st.write("Идет обучение...")
          if response.status_code == 200:
            result = response.json()
            st.success("Модель обучена")
            st.write(f"**ID модели:** {result['model_id']}")
            st.write(f"**Время обучения:** {result['execution_time']}")
            st.write(f"**Accuracy:** {result['accuracy']}")
            st.write(f"**Precision:** {result['precision']}")
            st.write(f"**Recall:** {result['recall']}")
            st.write(f"**F1 Score:** {result['f1']}")
          elif response.status_code == 404:
            st.error(response.json()['detail'])
          else:
            st.error(f"Ошибка при обучении: {response.status_code}")
        except Exception as e:
            st.error(f"Ошибка при обработке данных: {str(e)}")
    else:
        st.warning("Загрузите оба файла (тренировочный и тестовый).")



    st.title("Построение кривых обучения")

    if 'selected_model' not in st.session_state or not st.session_state['selected_model']:
          st.warning("Сначала выберите модель для обучения")
          if st.button("Перейти к выбору модели"):
            st.experimental_set_query_params(page="Информация про модели и данные")
          st.stop()

    st.write(f"Для построения кривых обучения вы используете модель: **{st.session_state['selected_model']}**")

    file = st.file_uploader("Загрузите датасет для обучения (файл формата Parquet)", type=["parquet"])

    st.header("Настройка гиперпараметров")
    random_state = st.number_input("random_state", value=42)
    max_iter = st.number_input("max_iter", value=1000, min_value=100, step=100)
    tol = st.number_input("tol", value=1e-4, format="%.1e")


    if st.button("Построить кривые обучения"):
      if file is not None:
        try:
          data = pd.read_parquet(file)
          if 'text' not in data.columns or 'author' not in data.columns:
            st.error("Датасет должен содержать колонки 'text' и 'author'.")
            st.stop()

          hyperparameters = {
                "hyperparameters": {
                    "random_state": random_state,
                    "max_iter": max_iter,
                    "tol": tol
                }
            }
          params = {'request': json.dumps(hyperparameters)}

          file.seek(0)
          files = {'file': file}
          response = requests.post(f"{API_URL}/LearningCurve", data=params, files=files)

          if response.status_code == 200:
            result = response.json()
            train_sizes = result['train_sizes']
            train_scores = result['train_scores']
            test_scores = result['test_scores']
            st.subheader("Кривые обучения")
            train_mean = [sum(scores) / len(scores) for scores in train_scores]
            test_mean = [sum(scores) / len(scores) for scores in test_scores]

            df = pd.DataFrame({
                    "Train Size": train_sizes * 2,
                    "Score": train_mean + test_mean,
                    "Type": ["Train"] * len(train_sizes) + ["Test"] * len(train_sizes)
                })

            fig = px.line(df, x="Train Size", y="Score", color="Type", title="Кривые обучения (Train/Test)")
            st.plotly_chart(fig, use_container_width=True)
            st.write("Детализация результатов:")
            st.write(pd.DataFrame({
                    "Train Size": train_sizes,
                    "Train Mean Accuracy": train_mean,
                    "Test Mean Accuracy": test_mean
                }))

          elif response.status_code == 404:
            st.error(response.json()['detail'])
          else:
            st.error(f"Ошибка при расчёте кривых обучения: {response.status_code}")
        except Exception as e:
          st.error(f"Ошибка при обработке данных: {str(e)}")
    else:
        st.warning("Загрузите файл для получения кривых обучения.")

    st.title("Обучение модели SVC")

    if 'selected_model' not in st.session_state or not st.session_state['selected_model']:
      st.warning("Сначала выберите активную модель для обучения")
    if st.button("Перейти к выбору модели"):
      st.experimental_set_query_params(page="Информация про модели и данные")
    st.stop()

    st.write(f"Выбранная активная модель: **{st.session_state['selected_model']}**")

    train_file = st.file_uploader("Загрузите датасет для дообучения (файл формата Parquet)", type=["parquet"])

    if st.button("Запустить обучение модели SVC"):
      if train_file is not None:
        try:
            data = pd.read_parquet(train_file)
            if 'X' not in data.columns or 'y' not in data.columns:
                st.error("Файл должен содержать колонки 'X' и 'y'")
                st.stop()
            train_file.seek(0)
            files = {'request_file': train_file}
            params = {'id': st.session_state['selected_model']}
            response = requests.post(f"{API_URL}/partial_fit", data=params, files=files)
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
