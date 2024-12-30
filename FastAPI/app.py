from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from contextlib import asynccontextmanager
import concurrent.futures
from http import HTTPStatus
from pydantic import BaseModel
from typing import List, Dict, Union, Any, Optional, Annotated
import joblib
import uvicorn
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import copy
import io

# Инициализация FastAPI приложения
app = FastAPI()

# Глобальная переменная для отслеживания активной модели
active_model_id = None

# Структура для хранения всех моделей
initial_models_list = {
    'model1': {
        'model': joblib.load('pipeline.joblib'),
        'description': '''pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
MaxAbsScaler(),
OneVsRestClassifier(LogisticRegression(solver='liblinear')))''',
    },
    'model2': {
        'model': joblib.load("ngram_naive_bayes.joblib"),
        'description': '''ngram_naive_bayes = make_pipeline(
CountVectorizer(ngram_range=(1, 2), max_features=500),
MultinomialNB())''',
    },
    'model3': {
        'model': joblib.load("tfidf_log_reg_standardized.joblib"),
        'description': '''tfidf_log_reg_standardized = make_pipeline(
TfidfVectorizer(max_features=333),
StandardScaler(with_mean=False),
LogisticRegression(random_state=42, max_iter=10000, solver='lbfgs'))''',
    },
    # Модель, которая поддерживает дообучение
    'model4': {
        'model': joblib.load("SGDClassifier.joblib"),
        'description': '''SGDpipe = make_pipeline(
TfidfVectorizer(max_features=700),
StandardScaler(with_mean=False),
SGDClassifier(max_iter=10000, tol=1e-3))''',
    }
}

# Список перенаученных моделей
refitted_models_list = {}

# Объединяем словари для упрощения проверки
all_models_list = {**initial_models_list, **refitted_models_list}

# Конекстный менеджер для управления жизненным циклом приложения
@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    global active_model_id
    print("Loading model...")
    active_model_id = 'model1'
    get_model(active_model_id)
    print("Model loaded.")
    yield
    print("Cleaning up...")

# Регистрируем lifespan для приложения
app.router.lifespan_context = lifespan

# Базовая модель запроса
class ModelRequestBase(BaseModel):
    pass

# Задаем различные модели запросов и ответов
class PredictItemRequest(ModelRequestBase):
    text: str

class PredictItemResponse(BaseModel):
    id: str
    author: str

class PredictItemsRequest(ModelRequestBase):
    texts: Dict[str, str]

class PredictItemsResponse(BaseModel):
    response: Dict[str, str]

class PredictItemsProbaResponse(BaseModel):
    response: Dict[str, Dict[str, float]]

class RefitModelRequest(ModelRequestBase):
    refitted_model_id: str
    hyperparameters: Dict[str, Any]

class RefitModelResponse(BaseModel):
    message: str
    metrics: Dict[str, float]
    train_sizes: List[Union[float, int]]
    train_scores: List[List[Union[float, int]]]
    test_scores: List[List[Union[float, int]]]

# Функция для получения модели по её идентификатору
def get_model(model_id: str) -> Dict[str, Any]:
    if model_id in initial_models_list:
        return initial_models_list[model_id]
    if model_id in refitted_models_list:
        return refitted_models_list[model_id]
    raise HTTPException(status_code=400, detail=f'Model with id "{model_id}" doesn\'t exist!')

# Асинхронная функция для предсказания одного элемента
async def predict(model: Any, text: str) -> str:
    text_series = pd.Series([text])
    return model.predict(text_series)[0]

# Асинхронная функция для предсказания вероятностей
async def predict_proba(model: Any, text: str) -> List[float]:
    text_series = pd.Series([text])
    return model.predict_proba(text_series)[0]

# Эндпоинт для получения списка моделей
@app.get('/ModelsList', response_model=List[Dict[str, str]])
async def models_list() -> List[Dict[str, str]]:
    models_info = []
    for model_name, model_info in all_models_list.items():
        info = {
            'name': model_name,
            'description': model_info.get('description', 'No description available'),
        }
        models_info.append(info)
    return models_info

# Эндпоинт для установки активной модели
@app.post("/setModel", status_code=HTTPStatus.OK)
async def set_model(
    request: ModelRequestBase,
    id: Annotated[str, Query(..., enum=list(all_models_list.keys()))]
) -> Dict[str, str]:
    global active_model_id
    if id not in all_models_list:
        raise HTTPException(status_code=400, detail=f'Model with id "{id}" doesn\'t exist!')
    active_model_id = id
    return {"message": f"Active model set to '{active_model_id}'"}

# Эндпоинт для предсказания одного элемента
@app.post("/PredictItem", response_model=PredictItemResponse, status_code=HTTPStatus.OK)
async def predict_item(request: PredictItemRequest) -> PredictItemResponse:
    model_id = active_model_id
    model = get_model(model_id)
    author_prediction = await predict(model['model'], request.text)
    return PredictItemResponse(id=model_id, author=author_prediction)

# Эндпоинт для предсказания одного элемента из файла
@app.post("/PredictItemFile", response_model=PredictItemResponse, status_code=HTTPStatus.OK)
async def predict_item_file(request: UploadFile = File()) -> PredictItemResponse:
    contents = await request.read()
    text_data = contents.decode('utf-8')
    model_id = active_model_id
    model = get_model(model_id)
    author_prediction = await predict(model['model'], text_data)
    return PredictItemResponse(id=model_id, author=author_prediction)

# Эндпоинт для предсказания вероятностей одного элемента
@app.post('/PredictItemProba', response_model=Dict[str, float], status_code=HTTPStatus.OK)
async def predict_item_proba(request: PredictItemRequest) -> Dict[str, float]:
    model_id = active_model_id
    model = get_model(model_id)
    probabilities = await predict_proba(model['model'], request.text)
    author_probas = dict(zip(model['model'].classes_, probabilities))
    return dict(sorted(author_probas.items(), key=lambda item: item[1], reverse=True))

# Эндпоинт для предсказания вероятностей одного элемента из файла
@app.post('/PredictItemProbaFile', response_model=Dict[str, float], status_code=HTTPStatus.OK)
async def predict_item_proba_file(request: UploadFile = File()) -> Dict[str, float]:
    contents = await request.read()
    model_id = active_model_id
    model = get_model(model_id)
    probabilities = await predict_proba(model['model'], contents.decode('utf-8'))
    author_probas = dict(zip(model['model'].classes_, probabilities))
    return dict(sorted(author_probas.items(), key=lambda item: item[1], reverse=True))

# Эндпоинт для предсказания нескольких элементов
@app.post('/PredictItems', response_model=PredictItemsResponse, status_code=HTTPStatus.OK)
async def predict_items(request: PredictItemsRequest) -> PredictItemsResponse:
    model_id = active_model_id
    model = get_model(model_id)
    texts_series = pd.Series(list(request.texts.values()))
    predictions = model['model'].predict(texts_series)
    author_predictions = dict(zip(request.texts.keys(), predictions))
    return PredictItemsResponse(response=author_predictions)

# Эндпоинт для предсказания вероятностей нескольких элементов
@app.post('/PredictItemsProba', response_model=PredictItemsProbaResponse, status_code=HTTPStatus.OK)
async def predict_items_proba(request: PredictItemsRequest) -> PredictItemsProbaResponse:
    model = get_model(active_model_id)
    texts_series = pd.Series(list(request.texts.values()))
    predictions_proba = model['model'].predict_proba(texts_series)
    author_prediction_probas = pd.DataFrame(predictions_proba, index=request.texts.keys(), columns=model['model'].classes_).T.to_dict()
    return PredictItemsProbaResponse(response=author_prediction_probas)

# # Асинхронная функция для чтения файла
# async def read_file(request_file: UploadFile):
#     contents = await request_file.read()
#     return pd.read_json(io.BytesIO(contents))

# # Функция для обучения модели
# def fit_model(model, X_train, y_train, X_test, y_test):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     metrics = {
#         'accuracy': accuracy_score(y_test, y_pred),
#         'precision': precision_score(y_test, y_pred, average='macro'),
#         'recall': recall_score(y_test, y_pred, average='macro'),
#         'f1': f1_score(y_test, y_pred, average='macro')
#     }
#     return metrics, model

# # Эндпоинт для переобучения модели
# @app.post("/refit_model_experiment", response_model=RefitModelResponse)
# async def refit_model_experiment(
#     id: Annotated[str, Form()],
#     refitted_model_id: Annotated[str, Form()],
#     hyperparameters: Annotated[str, Form()],
#     request_file: UploadFile = File(),
# ) -> RefitModelResponse:

#     if refitted_model_id in all_models_list:
#         raise HTTPException(status_code=400, detail=f'Model with id "{refitted_model_id}" already exists!')

#     model_data = get_model(id)
#     if not model_data:
#         raise HTTPException(status_code=404, detail=f'Model with id "{id}" not found!')

#     model = copy.deepcopy(model_data['model'])
#     data = await read_file(request_file)
#     X, y = data['X'], data['y']

#     hyperparameters = json.loads(hyperparameters)

#     try:
#         if hasattr(model, "steps") and model.steps[-1][0] == 'onevsrestclassifier':
#             model[-1].estimator.set_params(**hyperparameters)
#         else:
#             model[-1].set_params(**hyperparameters)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f'Error setting hyperparameters. Details: {e}')

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=17)

#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         future = executor.submit(fit_model, model, X_train, y_train, X_test, y_test)
#         try:
#             metrics, trained_model = future.result(timeout=10)
#         except concurrent.futures.TimeoutError:
#             future.cancel()
#             raise HTTPException(status_code=500, detail='Model training exceeded time limit of 10 seconds')
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f'Error during model fitting/predicting. Details: {e}')

#     try:
#         train_sizes, train_scores, test_scores = learning_curve(
#             model, X, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=2, scoring='accuracy', n_jobs=-1
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f'Error during fitting learning curve. Details: {e}')

#     train_sizes, train_scores, test_scores = map(np.ndarray.tolist, [train_sizes, train_scores, test_scores])

#     refitted_models_list[refitted_model_id] = {'model': trained_model}

#     response = RefitModelResponse(
#         message=f'Model with id "{id}" was successfully refitted with new id "{refitted_model_id}"',
#         metrics=metrics,
#         train_sizes=train_sizes,
#         train_scores=train_scores,
#         test_scores=test_scores
#     )
#     return response

# async def read_file(request_file):
#     content = await request_file.read()
#     data = np.load(content)
#     return {'X': data['text'], 'y': data['author']}

# Асинхронная функция для чтения файла
import pyarrow.parquet as pq

async def read_file(request_file: UploadFile):
    contents = await request_file.read()
    buffer = io.BytesIO(contents)
    data = pd.read_parquet(buffer, engine='pyarrow')
    return {'X': data['text'], 'y': data['author']}

@app.post("/partial_fit", response_model=Dict[str, str], status_code=HTTPStatus.OK)
async def partial_fit(
    id: Annotated[str, Form()],
    request_file: UploadFile = File()
) -> Dict[str, str]:
    model_id = id
    if model_id not in initial_models_list:
        raise HTTPException(status_code=400, detail=f'Model with id "{model_id}" doesn\'t exist!')

    pipeline = initial_models_list[model_id]['model']

    # Считываем данные для обучения
    data = await read_file(request_file)
    X_train, y = data['X'], data['y']

    # Извлекаем TfidfVectorizer и SGDClassifier из пайплайна
    vectorizer = pipeline.named_steps['tfidfvectorizer']
    model = pipeline.named_steps['sgdclassifier']

    # Преобразуем текстовые данные в tf-idf представление
    X_tfidf = vectorizer.transform(X_train)
    
    # Загрузка классов из файла
    loaded_classes = joblib.load('authors.pkl')

    # Обучаем модель на новых данных
    try:
        model.partial_fit(X_tfidf, y, classes=loaded_classes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error during partial fitting. Details: {e}')
    
    # Обновляем модель в пайплайне
    pipeline.named_steps['sgdclassifier'] = model
    joblib.dump(pipeline, f'{model_id}.joblib')
    
    return {"message": f"Model with id '{model_id}' successfully updated with new data."}

# Запуск приложения
if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)