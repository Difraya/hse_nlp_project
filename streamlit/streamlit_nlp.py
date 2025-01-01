import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import re
import json
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download("punkt")

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
    pos = [i[1] for i in pos_tag(word_only)]
    pos_count = Counter(pos)
    return pos_count

if choice == "Пользовательская часть":
    st.title("Пользовательская часть")
    # обработка фрагмента текста
    if st.button("Ввести свой текст"):
      txt = st.text_area("Введите текст:", "")

      if txt:
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
  
        if 'selected_model' not in st.session_state:
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
    #      w = get_top_words(txt)
    #      x = [i[0] for i in w]
      #    y = [i[1] for i in w]
      #    fig = px.bar(x=x, y=y, title="Топ слов")
        #  st.plotly_chart(fig, use_container_width=True)
          res = requests.post(f"{API_URL}/PredictItemProba", json={"text": txt})
          if r.status_code == 200:
            pred = r.json()
            st.success("Самые близкие вам авторы в процентах:")
            for author, proba in pred.items():
              st.write(f"{author}: {proba:4f}")
        else:
          st.warning("Введите текст для предсказания")

    # обработка датасета
    if st.button("Загрузить свой файл"):
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
        # сюда вставить визуализации
      
      if st.button("Сделать предсказания по файлу"):

        if 'selected_model' not in st.session_state:
          st.warning("Сначала выберите модель для предсказания")
          if st.button("Перейти к выбору модели"):
            st.experimental_set_query_params(page="Информация про модели и данные")
          st.stop()
        
        st.write(f"Для предсказаний вы используете модель: **{st.session_state['selected_model']}**")

        # тут дописать ручки из апи с предсказаниями

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
    response = (f"{API_URL}/ModelsList")
    if response.status_code == 200:
        models = response.json()
        model_names = [model['name'] for model in models]
        selected_model = st.selectbox("Выберите модель", model_names)

        if st.button(f"Установить модель {selected_model}"):
            res = requests.post(f"{API_URL}/setModel", params={"id": selected_model})
            if res.status_code == 200:
              st.session_state['selected_model'] = selected_model
              st.success(f"Активная модель установлена: {selected_model}")
            elif res.status_code == 400:
              st.error(f'Модель {selected_model} не существует!')
            else:
              st.error("Не удалось установить модель.")
    else:
        st.error("Нет доступных моделей.")

  #  response = requests.get(f"{API_URL}/ModelsList")
  #  st.write("Авторы в обучающем датасете (пример)")
  #  fig1 = px.bar(x=["Автор1", "Автор2", "Автор3"], y=[100, 80, 50], title="Количество текстов")
  #  st.plotly_chart(fig1, use_container_width=True)
  #  st.write("Пример EDA")
  #  fig2 = go.Figure(data=go.Scatter(x=[1,2,3,4,5], y=[10,4,6,3,8], mode='lines+markers'))
  #  st.plotly_chart(fig2, use_container_width=True)

elif choice == "Обучи свою модель":
    st.title("Обучи свою модель")
    data_file = st.file_uploader("Загрузите датасет для обучения")
    if st.button("Запуск обучения"):
        st.write("Идет обучение...")
        st.write("Модель обучена")
    st.write("Аналитика")
    fig3 = px.line(x=[1,2,3,4], y=[0.6,0.7,0.8,0.95], title="Кривая обучения")
    st.plotly_chart(fig3, use_container_width=True)
    st.write("Инференс новой моделью")
    txt_new = st.text_area("Введите текст")
    if st.button("Предсказать новой моделью"):
        st.write("Предсказан автор: ...")
