import streamlit as st
import pandas as pd
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# Функция для предсказания
def predict(text):
    url = f"{BASE_URL}/PredictItem"
    headers = {"Content-Type": "application/json"}
    data = {"text": text}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

# Интерфейс приложения
st.title("Машинное обучение и инференс")

# Загрузка датасета
st.sidebar.header("Загрузка датасета")
uploaded_file = st.sidebar.file_uploader("Загрузите ваш датасет в формате Parquet", type=["parquet"])
if uploaded_file:
    data = pd.read_parquet(uploaded_file)
    st.write("Данные загружены:")
    st.write(data.head())

# Просмотр списка моделей
if st.sidebar.button("Получить список моделей"):
    response = requests.get(f"{BASE_URL}/ModelsList")
    models = response.json()
    for model in models:
        st.sidebar.write(f"Модель: {model['name']}, Описание: {model['description']}")

# Выбор модели
model_id = st.sidebar.selectbox("Выберите модель:", ["model1", "model2", "model3", "model4"])

# Установка активной модели
if st.sidebar.button("Установить активную модель"):
    response = requests.post(f"{BASE_URL}/setModel?id={model_id}")
    st.sidebar.write(response.json()["message"])

# Предсказание
st.subheader("Предсказание текстов")
text_input = st.text_area("Введите текст для предсказания:")
if st.button("Предсказать"):
    if text_input:
        prediction = predict(text_input)
        st.write(f"Предсказанный автор: {prediction['author']}")

# Обучение модели
if st.sidebar.button("Обучить модель"):
    if uploaded_file:
        train_file = uploaded_file  # предположим, что можно использовать один и тот же файл для обучения и тестирования
        test_file = uploaded_file
        hyperparameters = {}  # гиперпараметры можно добавить как отдельные параметры

        # Эндпоинт для обучения
        train_url = f"{BASE_URL}/TrainModel"
        files = {
            "train_file": train_file.getvalue(),
            "test_file": test_file.getvalue(),
        }
        data = {"hyperparameters": hyperparameters}
        response = requests.post(train_url, files=files, data=data)

        # Результат обучения
        if response.status_code == 200:
            result = response.json()
            st.write(f"Модель {result['model_id']} успешно обучена.")
            st.write(f"Точность: {result['accuracy']}")
            st.write(f"Время выполнения: {result['execution_time']}")
        else:
            st.write(f"Ошибка: {response.json()['detail']}")
    else:
        st.write("Пожалуйста, загрузите датасет для обучения.")

# Информация о кривых обучения
st.subheader("Кривые обучения:")
# Здесь можно добавить логику для отображения графиков кривых обучения, сохраняя их в FastAPI после обучения

st.sidebar.header("Информация о модели")
# Здесь можно добавить возможность отображать более детальные результаты и метрики модели после обучения
