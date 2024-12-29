from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from http import HTTPStatus
from pydantic import BaseModel
from typing import List, Dict, Union, Optional, Any
import joblib
import uvicorn
import pandas as pd
import numpy as np
import json 
import sklearn 
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
import copy 
import io


# Импортируем модели
model1 = joblib.load('pipeline.pkl')
model2 = joblib.load("ngram_naive_bayes.pkl")

# Создаем хранилище дефолтных моделей
initial_models_list = {'model1': model1, 'model2': model2}

# Создаем хранилище переобученных моделей
refitted_models_list = {}

# Инициализируем приложение
app = FastAPI()

# Запрос входных данных для одного текста
class PredictItemRequest(BaseModel):
    text: str
    model_id: str = 'model1'

# Ответ на предсказание 1 автора
class PredictItemResponse(BaseModel):
    author: str

# Запрос входных данных для нескольких текстов
class PredictItemsRequest(BaseModel):
    texts: Dict[str, str]
    model_id: str = 'model1'


# Модель выходных данных для предсказания нескольких текстов
class PredictItemsResponse(BaseModel):
    response: Dict[Union[str, float, int], str]


# Модель выходных данных для предсказания вероятностей для нескольких текстов
class PredictItemsProbaResponse(BaseModel):
    response: Dict[Union[str, float, int], Dict[str, Union[float, int]]]


class RefitModelRequest(BaseModel):
    model_id: str = 'model1'
    refitted_model_id: str
    hyperparameters: Dict[str, Any]


class RefitModelResponse(BaseModel):
    message: str 
    metrics: Dict[str, float]
    train_sizes: List[Union[float, int]]
    train_scores: List[List[Union[float, int]]]
    test_scores: List[List[Union[float, int]]]



@app.post("/PredictItem", response_model=PredictItemResponse, status_code=HTTPStatus.OK) 
async def PredictItem(request: PredictItemRequest):
    # Получаем текст на вход и добавляем его в Series, чтобы можно было передать модели
    text = request.text
    text = pd.Series([text])
    # Выбираем модель
    if request.model_id in initial_models_list.keys():
        model = initial_models_list[request.model_id]
    elif request.model_id in refitted_models_list.keys():
        model = refitted_models_list[request.model_id]
    else:
        raise HTTPException(status_code=400, detail=f'Model with id "{request.model_id}" doesn\'t exist!')
    # Исходим из того, что в pipeline уже есть предобработка (стоп-слова, токенизация и т.п.)
    try:
        author_prediction = model.predict(text)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Internal server error. Details: "{e}"')
    # Формируем ответ
    response = PredictItemResponse(author=author_prediction)

    return response


@app.post("/PredictItemFile", response_model=PredictItemResponse, status_code=HTTPStatus.OK) 
async def PredictItemFile(request: UploadFile = File(), model_id: str = Form(default='model1')):
    # Получаем файл и обрабатываем его
    text = await request.read()
    text = pd.Series([text])
    # Выбираем модель
    if model_id in initial_models_list.keys():
        model = initial_models_list[model_id]
    elif model_id in refitted_models_list.keys():
        model = refitted_models_list[model_id]
    else:
        raise HTTPException(status_code=400, detail=f'Model with id "{model_id}" doesn\'t exist!')
    # Исходим из того, что в pipeline уже есть предобработка (стоп-слова, токенизация и т.п.)
    try:
        author_prediction = model.predict(text)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error. Details: '{e}'")
    # Формируем ответ
    response = PredictItemResponse(author=author_prediction)

    return response


@app.post('/PredictItemProba', response_model=Dict[str, float], status_code=HTTPStatus.OK)
async def PredictItemProba(request: PredictItemRequest):
    # Получаем текст на вход и добавляем его в Series, чтобы можно было передать модели
    text = request.text
    text = pd.Series([text])
    # Выбираем модель
    if request.model_id in initial_models_list.keys():
        model = initial_models_list[request.model_id]
    elif request.model_id in refitted_models_list.keys():
        model = refitted_models_list[request.model_id]
    else:
        raise HTTPException(status_code=400, detail=f'Model with id "{request.model_id}" doesn\'t exist!')
    # Исходим из того, что в pipeline уже есть предобработка (стоп-слова, токенизация и т.п.)
    try:
        author_prediction = model.predict_proba(text)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error. Details: '{e}'")
    # Сортируем вероятности
    author_probas = dict(zip(model.classes_, author_prediction))
    author_probas = {k: v for k, v in sorted(author_probas .items(), key=lambda item: item[1], reverse=True)}

    return author_probas



@app.post('/PredictItemProbaFile', response_model=Dict[str, float], status_code=HTTPStatus.OK)
async def PredictItemProbaFile(request: UploadFile = File(), model_id: str = Form(default='model1')):
    # Получаем файл и обрабатываем его
    text = await request.read()
    text = pd.Series([text])
    # Выбираем модель
    if model_id in initial_models_list.keys():
        model = initial_models_list[model_id]
    elif model_id in refitted_models_list.keys():
        model = refitted_models_list[model_id]
    else:
        raise HTTPException(status_code=400, detail=f'Model with id "{model_id}" doesn\'t exist!')
    # Исходим из того, что в pipeline уже есть предобработка (стоп-слова, токенизация и т.п.)
    try:
        author_prediction = model.predict_proba(text)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error. Details: '{e}'")
    # Сортируем вероятности
    author_probas = dict(zip(model.classes_, author_prediction))
    author_probas = {k: v for k, v in sorted(author_probas .items(), key=lambda item: item[1], reverse=True)}

    return author_probas


@app.post('/PredictItems', response_model=PredictItemsResponse, status_code=HTTPStatus.OK)
async def PredictItems(request: PredictItemsRequest):
    # Приводим тексты к формату, который принимает модель
    texts = pd.Series(request.texts).T
    # Выбираем модель
    if request.model_id in initial_models_list.keys():
        model = initial_models_list[request.model_id]
    elif request.model_id in refitted_models_list.keys():
        model = refitted_models_list[request.model_id]
    else:
        raise HTTPException(status_code=400, detail=f'Model with id "{request.model_id}" doesn\'t exist!')
    # Делаем предсказание. 
    try:
        pred = model.predict(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error. Details: '{e}'")
    # Соединяем предсказания с лейблами переданных текстов
    author_predictions = dict(zip(texts.index, pred))
    # Создаем модель с ответами
    response = PredictItemsResponse(response=author_predictions)

    return response


@app.post('/PredictItemsFile', response_model=PredictItemsResponse, status_code=HTTPStatus.OK)
async def PredictItemsFile(request: UploadFile = File(), model_id: str = Form(default='model1')):
    # Читаем содержимое файла
    contents = await request.read()
    # Преобразуем файл в json
    json_data = json.loads(contents)
    # Преобразование JSON в Series
    texts = pd.Series(json_data)
    # Выбираем модель
    if model_id in initial_models_list.keys():
        model = initial_models_list[model_id]
    elif model_id in refitted_models_list.keys():
        model = refitted_models_list[model_id]
    else:
        raise HTTPException(status_code=400, detail=f'Model with id "{model_id}" doesn\'t exist!')
    # Делаем предсказание. 
    try:
        pred = model.predict(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error. Details: '{e}'")
    # Соединяем предсказания с лейблами переданных текстов
    author_predictions = dict(zip(texts.index, pred))
    # Создаем модель с ответами
    response = PredictItemsResponse(response=author_predictions)

    return response


@app.post('/PredictItemsProba', response_model=PredictItemsProbaResponse, status_code=HTTPStatus.OK)
async def PredictItemsProba(request: PredictItemsRequest):
    # Преобразовываем текст в формат, который можно передать модели
    texts = pd.Series(request.texts).T
    # Выбираем модель
    if request.model_id in initial_models_list.keys():
        model = initial_models_list[request.model_id]
    elif request.model_id in refitted_models_list.keys():
        model = refitted_models_list[request.model_id]
    else:
        raise HTTPException(status_code=400, detail=f'Model with id "{request.model_id}" doesn\'t exist!')
    # Делаем предсказание
    try:
        preds = model.predict_proba(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Internal server error. Details: "{e}"')
    # Приводим к нужному типу
    author_prediction_probas = pd.DataFrame(preds, index=texts.index, columns=model.classes_).T.to_dict()
    # Сформируем ответ
    response = PredictItemsProbaResponse(response=author_prediction_probas)

    return response


@app.post('/PredictItemsProbaFile', response_model=PredictItemsProbaResponse, status_code=HTTPStatus.OK) 
async def PredictItemsProbaFile(request: UploadFile = File(), model_id: str = Form(default='model1')):
    # Преобразовываем текст в формат, который можно передать модели. Читаем содержимое файла
    contents = await request.read()
    # Преобразуем файл в json
    json_data = json.loads(contents)
    # Преобразование JSON в Series
    texts = pd.Series(json_data)
    # Выбираем модель
    if model_id in initial_models_list.keys():
        model = initial_models_list[model_id]
    elif model_id in refitted_models_list.keys():
        model = refitted_models_list[model_id]
    else:
        raise HTTPException(status_code=400, detail=f'Model with id "{model_id}" doesn\'t exist!')
    # Делаем предсказание
    try:
        preds = model.predict_proba(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Internal server error. Details: "{e}"')
    # Приводим данные к нужному типу
    author_prediction_probas = pd.DataFrame(model.predict_proba(texts), index=texts.index, columns=model.classes_).T.to_dict()
    # Сформируем ответ
    response = PredictItemsProbaResponse(response=author_prediction_probas)
    
    return response 



@app.post('/RefitModelExperiment', status_code=HTTPStatus.OK)
async def RefitModelExperiment(request_file: UploadFile = File(), model_id: str = Form(), refitted_model_id: str = Form(), hyperparameters: str = Form()):
    # Проверка на дублирующиеся имена 
    if (refitted_model_id in initial_models_list.keys()) or (refitted_model_id in refitted_models_list.keys()):
        raise HTTPException(status_code=400, detail=f'Model with id "{refitted_model_id}" already exists!') 
    # Проверка на наличие id 
    if (model_id in initial_models_list.keys()) or (model_id in refitted_models_list.keys()):
        # Инициализируем модель
        model = copy.deepcopy(initial_models_list[model_id]) 
    else:   
        raise HTTPException(status_code=400, detail=f'Model with id "{model_id}" doesn\'t exist!')

    # Приводим данные к нужному типу
    contents = await request_file.read()
    data = pd.read_json(io.BytesIO(contents))
    X, y = data['X'], data['y']

    # Передаем модели гиперпараметры. Если гиперпараметры неверны, обрабатываем исключение
    try:
        hyperparameters = json.loads(hyperparameters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during parsing hyperparameters. Details: {e}")

    try:
        if model.steps[-1][0] == 'onevsrestclassifier':
            model[-1].estimator.set_params(**hyperparameters)
        else:
            model[-1].set_params(**hyperparameters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Internal error during passing params. Details: {e}')

    # Выделим тестовые и тренировочные данные (нужно в дальнейшем для валидации)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=17)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during train, test split. Details: {e}")

    # Рассчитываем данные для learning_curve
    try:
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=2, scoring='accuracy', n_jobs=-1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error during fitting learning curve. Details: {e}') 

    # Приводим результаты learing curve к нужному типу
    train_sizes, train_scores, test_scores = train_sizes.tolist(), train_scores.tolist(), test_scores.tolist()
    
    # Пытемся обучить модель: 
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Internal server error during fitting. Details: "{e}"')

    # Пытаемся следать предсказания 
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error during prediction. Details: {e}')


    # Рассчитываем метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro') 
    # Записываем метрики в словарь
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    # Сохраняем модель
    refitted_models_list[refitted_model_id] = model 
    
    # Возвращаем сообщение 
    response = RefitModelResponse(message=f'Model with id "{model_id}" was successfully refitted with new id "{refitted_model_id}"', metrics=metrics, train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores)
    
    return response



@app.get('/ModelsList', response_model=List[str])
async def ModelsList():
    model_ids = list(initial_models_list.keys()) + list(refitted_models_list.keys())
    return model_ids


if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)