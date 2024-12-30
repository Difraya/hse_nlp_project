from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Body
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
from http import HTTPStatus
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Any, Optional, Annotated
import joblib
import uvicorn
import pandas as pd
import json
import time
import asyncio
import io
from ngram_naive_bayes import model2
from tfidf_log_reg_standardized import model3
from SGDClassifier import model4

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


# Функция для получения модели по её идентификатору
def get_model(model_id: str) -> Dict[str, Any]:
    if model_id in initial_models_list:
        return initial_models_list[model_id]

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
    for model_name, model_info in initial_models_list.items():
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
    id: Annotated[str, Query(..., enum=list(initial_models_list.keys()))]
) -> Dict[str, str]:
    global active_model_id
    if id not in initial_models_list:
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


# Асинхронная функция для чтения файла
async def read_parquet_file(upload_file: UploadFile):
    contents = await upload_file.read()
    buffer = io.BytesIO(contents)
    data = pd.read_parquet(buffer, engine='pyarrow')

    return data


# Словарь, сопоставляющий имена моделей с функциями
model_functions = {
    "model2": model2,
    "model3": model3,
    "model4": model4
}

# Эндпоинт для обучения активной модели
@app.post("/train_model")
async def train_model(
    request: str = Form('{"hyperparameters": {"random_state": 42, \
"max_iter": 1000, "tol": 1e-4}}',
                        description="Dictionary of hyperparameters as JSON string"),
    train_file: UploadFile = File(..., description="Training dataset in Parquet format"),
    test_file: UploadFile = File(..., description="Testing dataset in Parquet format"),
):
    model_id = active_model_id

    # Проверка ID модели
    if model_id == 'model1':
        raise HTTPException(status_code=404, detail="Обучение model1 занимает слишком много времени, пожалуйста, активируйте другую модель из списка")
    elif model_id not in model_functions:
        raise HTTPException(status_code=404, detail="Model not found")

    # Чтение данных из файлов
    train_data = await read_parquet_file(train_file)
    test_data = await read_parquet_file(test_file)

    X_train, y_train = train_data['text'], train_data['author']
    X_test, y_test = test_data['text'], test_data['author']

    # Получение функции обучения модели
    train_function = model_functions.get(model_id)

    try:
        request_data = json.loads(request)
        hyperparameters = request_data.get('hyperparameters', {})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in 'request'")

    start_time = time.perf_counter()
    try:
        accuracy = train_function(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            **hyperparameters
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error training model: {e}')

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    execution_time = str(round(execution_time, 2))

    return {
        "model_id": model_id,
        "accuracy": str(accuracy),
        "execution_time": f"{execution_time} seconds"
    }


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