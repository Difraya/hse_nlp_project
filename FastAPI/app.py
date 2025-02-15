from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import List, Dict, Union, Any
import json
import time
import io
import os
import re
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from pydantic import BaseModel
import joblib
import uvicorn
import pandas as pd
from models.ngram_naive_bayes import MNB_BoW_NB2
from models.tfidf_log_reg_standardized import Logreg_TFIDF
from models.SGDClassifier import SGDClassifier

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

# Инициализация FastAPI приложения
app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Позволяет всем доменам отправлять запросы
    allow_credentials=True,
    allow_methods=["*"],  # Позволяет использовать любые методы
    allow_headers=["*"],  # Позволяет использовать любые заголовки
)
logger.info("Добавлен CORSMiddleware.")

# Глобальная переменная для отслеживания активной модели
active_model_id = None

# Структура для хранения всех моделей
initial_models_list = {
'Logreg_TFIDF_NB2': {
  'model': joblib.load('models/pipeline.joblib'),
  'description': '''
Эта модель использует конвейер (pipeline), состоящий
из следующих шагов:
1. TF-IDF векторизация текста с биграммами.
2. Масштабирование признаков с применением MaxAbsScaler.
3. Классификация с помощью OneVsRest подхода и
логистической регрессии (solver='liblinear').
Точность модели (accuracy): 0.8043''',
    },
'MNB_BoW_NB2': {
  'model': joblib.load("models/ngram_naive_bayes.joblib"),
  'description': '''
Данная модель использует наивный байесовский классификатор. 
Конвейер включает:
1. Преобразование текста в числовые векторы с
использованием CountVectorizer с биграммами
и ограничением в 500 признаков.
2. Классификацию с помощью MultinomialNB.
Точность модели (accuracy): 0.7391''',
    },
'Logreg_TFIDF': {
  'model': joblib.load("models/tfidf_log_reg_standardized.joblib"),
  'description': '''
Эта модель построена на основе логистической регрессии
и включает следующие шаги:
1. Преобразование текста в TF-IDF признаки,
ограниченное 333 признаками.
2. Стандартизация данных с использованием
StandardScaler (без вычитания среднего).
3. Классификация с помощью логистической регрессии
с установленным random_state для воспроизводимости.
Точность модели (accuracy): 0.7826''',
    },
'SGDClassifier': {
  'model': joblib.load("models/SGDClassifier.joblib"),
  'description': '''
Эта модель поддерживает дообучение
и включает следующие шаги в конвейере:
1. Преобразование текста в TF-IDF признаки,
ограниченное 700 признаками.
2. Стандартизация данных с использованием
StandardScaler (без вычитания среднего).
3. Классификация с использованием SGDClassifier,
который обучается на нескольких итерациях.
Точность модели (accuracy): 0.6522''',
    }
}

# Конекстный менеджер для управления жизненным циклом приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Асинхронный менеджер контекста для управления жизненным циклом
    приложения FastAPI.
    Этот менеджер контекста выполняет начальную загрузку модели
    перед запуском серверного приложения FastAPI и обеспечивает очистку
    ресурсов после завершения работы приложения.

    Параметры:
    - app (FastAPI): Экземпляр приложения FastAPI, для которого
    определяется период жизненного цикла.

    Действия:
    - Устанавливает глобальный идентификатор `active_model_id`
    для активной модели.
    - Загружает модель с помощью функции `get_model`.
    - Регистрирует начало и завершение процессов загрузки модели
    и очистки ресурсов.
    """

    global active_model_id
    logger.info("Начало загрузки модели...")
    active_model_id = 'Logreg_TFIDF_NB2'
    get_model(active_model_id)
    logger.info("Модель загружена.")
    yield
    logger.info("Очистка ресурсов...")

# Регистрируем lifespan для приложения
app.router.lifespan_context = lifespan

# Задаем различные модели запросов и ответов
class PredictItemRequest(BaseModel):
    text: str


class PredictItemResponse(BaseModel):
    mod_id: str
    author: str


class PredictItemsRequest(BaseModel):
    texts: Dict[str, str]


class PredictItemsResponse(BaseModel):
    response: Dict[str, str]


class PredictItemsProbaResponse(BaseModel):
    response: Dict[str, Dict[str, float]]


class TrainModelResponse(BaseModel):
    mod_id: str
    execution_time: str
    accuracy: str
    precision: str
    recall: str
    f1: str
    train_sizes: List[Union[float, int]]
    train_scores_mean: List[Union[float, int]]
    test_scores_mean: List[Union[float, int]]


# Функция для получения модели по её идентификатору
def get_model(mod_id: str) -> Dict[str, Any]:
    """
    Получает модель по её идентификатору из списка начальных моделей.
    Параметры:
        model_id (str): Идентификатор модели.
    Возвращает:
        Dict[str, Any]: Словарь, содержащий модель и её описание.
    Исключения:
        HTTPException: Вызывается, если модель с таким идентификатором
        не существует.
    """
    if mod_id in initial_models_list:
        return initial_models_list[mod_id]

    logger.error('Модель с id "%s" не существует!', mod_id)

    raise HTTPException(status_code=400,
                        detail=f'Model with id "{mod_id}" doesn\'t exist!')


# Асинхронная функция для предсказания одного элемента
async def predict(model: Any, text: str) -> str:
    """
    Делает предсказание на основе текста, используя заданную модель.
    Параметры:
        model (Any): Модель, используемая для предсказания.
        text (str): Текст, для которого необходимо сделать предсказание.
    Возвращает:
        str: Предсказанный результат модели.
    """
    text_series = pd.Series([text])

    return model.predict(text_series)[0]


# Асинхронная функция для предсказания вероятностей
async def predict_proba(model: Any, text: str) -> List[float]:
    """
    Делает вероятностное предсказание на основе текста,
    используя заданную модель
    Параметры:
        model (Any): Модель, используемая для предсказания.
        text (str): Текст, для которого необходимо сделать
        предсказание вероятностей.
    Возвращает:
        List[float]: Вероятности принадлежности к каждому из классов.
    """
    text_series = pd.Series([text])

    return model.predict_proba(text_series)[0]


# Эндпоинт для получения AJAX-запросов
@app.get("/")
async def read_root():
    """
    Возвращает главный HTML файл (index.html) для корневого эндпоинта
    Возвращает:
        FileResponse: HTML файл для отображения.
    """

    return FileResponse("index.html")


# Эндпоинт для получения списка моделей
@app.get('/ModelsList', response_model=List[Dict[str, str]])
async def models_list() -> List[Dict[str, str]]:
    """
    Возвращает список всех доступных моделей с их описаниями
    Возвращает:
        List[Dict[str, str]]: Список словарей с именами моделей
        и их описаниями.
    """
    models_info = []
    for model_name, model_info in initial_models_list.items():
        info = {
            'name': model_name,
            'description': model_info.get('description',
                                          'No description available'),
        }
        models_info.append(info)

    logger.info("Возвращен список моделей.")

    return models_info


# Эндпоинт для установки активной модели
@app.post("/setModel", status_code=HTTPStatus.OK)
async def set_model(mod_id: str) -> Dict[str, str]:
    """
    Устанавливает активную модель по её идентификатору.
    Параметры:
        mod_id (Annotated[str]): Идентификатор модели для установки.
    Возвращает:
        Dict[str, str]: Сообщение об успешной установке модели.
    Исключения:
        HTTPException: Вызывается, если модель с данным
        идентификатором не найдена.
    """
    global active_model_id
    if mod_id not in initial_models_list:
        logger.error('Модель с id "%s" не существует!', mod_id)
        raise HTTPException(
            status_code=400,
            detail=f'Model with id "{mod_id}" doesn\'t exist!')

    active_model_id = mod_id
    logger.info("Установлена активная модель '%s'", active_model_id)

    return {"message": f"Active model set to '{active_model_id}'"}


# Эндпоинт для предсказания одного элемента
@app.post("/PredictItem", response_model=PredictItemResponse,
          status_code=HTTPStatus.OK)
async def predict_item(request: PredictItemRequest) -> PredictItemResponse:
    """
    Эндпоинт для предсказания одного элемента текста.
    Параметры:
        request (PredictItemRequest): Запрос, содержащий текст
        для предсказания.
    Возвращает:
        PredictItemResponse: Ответ с идентификатором модели
        и предсказанным автором.
    """
    mod_id = active_model_id
    model = get_model(mod_id)
    author_prediction = await predict(model['model'], request.text)

    logger.info('Предсказание сделано для одного элемента с \
                 использованием модели %s.', mod_id)

    return PredictItemResponse(mod_id=mod_id, author=author_prediction)


# Эндпоинт для предсказания одного элемента из файла
@app.post("/PredictItemFile", response_model=PredictItemResponse,
          status_code=HTTPStatus.OK)
async def predict_item_file(request: UploadFile = File()) \
                             -> PredictItemResponse:
    """
    Эндпоинт для предсказания одного элемента текста из загруженного файла.
    Параметры:
        request (UploadFile): Файл с текстом для предсказания.
    Возвращает:
        PredictItemResponse: Ответ с идентификатором модели
        и предсказанным автором.
    """
    contents = await request.read()
    text_data = contents.decode('utf-8')
    mod_id = active_model_id
    model = get_model(mod_id)
    author_prediction = await predict(model['model'], text_data)

    logger.info('Предсказание сделано для одного элемента из файла с \
                 использованием модели %s.', mod_id)

    return PredictItemResponse(mod_id=mod_id, author=author_prediction)


# Эндпоинт для предсказания вероятностей одного элемента
@app.post('/PredictItemProba', response_model=Dict[str, float],
          status_code=HTTPStatus.OK)
async def predict_item_proba(request: PredictItemRequest) -> Dict[str, float]:
    """
    Эндпоинт для предсказания вероятностей принадлежности одного
    элемента к классам.
    Параметры:
        request (PredictItemRequest): Запрос, содержащий текст для
        предсказания вероятностей.
    Возвращает:
        Dict[str, float]: Словарь с классами и их вероятностями.
    """
    mod_id = active_model_id
    model = get_model(mod_id)

    if mod_id == 'SGDClassifier':
        # Выдаем сообщение об ошибке, если активна SGDClassifier
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Модель 'SGDClassifier' не может предсказывать вероятности. "
                   "Пожалуйста, выберите другую модель."
        )

    probabilities = await predict_proba(model['model'], request.text)
    author_probas = dict(zip(model['model'].classes_, probabilities))

    logger.info('Вероятностное предсказание сделано для одного элемента \
                 с использованием модели %s.', mod_id)

    return dict(sorted(author_probas.items(),
                       key=lambda item: item[1], reverse=True))


# Эндпоинт для предсказания вероятностей одного элемента из файла
@app.post('/PredictItemProbaFile', response_model=Dict[str, float],
          status_code=HTTPStatus.OK)
async def predict_item_proba_file(request: UploadFile = File()) \
                                  -> Dict[str, float]:
    """
    Эндпоинт для предсказания вероятностей принадлежности текста
    из файла к классам.
    Параметры:
        request (UploadFile): Файл с текстом для предсказания вероятностей.
    Возвращает:
        Dict[str, float]: Словарь с классами и их вероятностями.
    """
    contents = await request.read()
    mod_id = active_model_id
    model = get_model(mod_id)

    if mod_id == 'SGDClassifier':
        # Выдаем сообщение об ошибке, если активна SGDClassifier
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Модель 'SGDClassifier' не может предсказывать вероятности. "
                   "Пожалуйста, выберите другую модель."
        )

    probabilities = await predict_proba(model['model'],
                                        contents.decode('utf-8'))
    author_probas = dict(zip(model['model'].classes_, probabilities))

    logger.info('Вероятностное предсказание сделано для одного элемента из \
                файла с использованием модели %s.', mod_id)

    return dict(sorted(author_probas.items(),
                       key=lambda item: item[1], reverse=True))


# Эндпоинт для предсказания нескольких элементов
@app.post('/PredictItems', response_model=PredictItemsResponse,
          status_code=HTTPStatus.OK)
async def predict_items(request: PredictItemsRequest) -> PredictItemsResponse:
    """
    Эндпоинт для предсказания нескольких элементов текста.
    Параметры:
        request (PredictItemsRequest): Запрос, содержащий тексты
        для предсказания.
    Возвращает:
        PredictItemsResponse: Ответ с предсказанными авторами для
        каждого текста.
    """
    mod_id = active_model_id
    model = get_model(mod_id)
    texts_series = pd.Series(list(request.texts.values()))
    predictions = model['model'].predict(texts_series)
    author_predictions = dict(zip(request.texts.keys(), predictions))

    logger.info('Предсказания сделаны для нескольких элементов \
                 с использованием модели %s.', mod_id)

    return PredictItemsResponse(response=author_predictions)


# Эндпоинт для предсказания вероятностей нескольких элементов
@app.post('/PredictItemsProba', response_model=PredictItemsProbaResponse,
          status_code=HTTPStatus.OK)
async def predict_items_proba(request: PredictItemsRequest) \
                              -> PredictItemsProbaResponse:
    """
    Эндпоинт для предсказания вероятностей принадлежности нескольких
    текстов к классам.
    Параметры:
        request (PredictItemsRequest): Запрос, содержащий тексты
        для предсказания вероятностей.
    Возвращает:
        PredictItemsProbaResponse: Ответ с вероятностями
        для каждого текста и класса.
    """
    if active_model_id == 'SGDClassifier':
        # Выдаем сообщение об ошибке, если активна SGDClassifier
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Модель 'SGDClassifier' не может предсказывать вероятности. "
                   "Пожалуйста, выберите другую модель."
        )

    model = get_model(active_model_id)
    texts_series = pd.Series(list(request.texts.values()))
    pred_proba = model['model'].predict_proba(texts_series)
    pred_probas = pd.DataFrame(pred_proba,
                               index=request.texts.keys(),
                               columns=model['model'].classes_).T.to_dict()

    logger.info('Вероятностные предсказания сделаны для нескольких элементов \
                с использованием модели %s.', active_model_id)

    return PredictItemsProbaResponse(response=pred_probas)


# Асинхронная функция для чтения файла
async def read_parquet_file(upload_file: UploadFile):
    """
    Асинхронное чтение parquet файла и преобразование его в DataFrame.
    Параметры:
        upload_file (UploadFile): Загруженный файл в формате parquet.
    Возвращает:
        DataFrame: Данные, содержащиеся в parquet файле.
    """
    contents = await upload_file.read()
    buffer = io.BytesIO(contents)
    data = pd.read_parquet(buffer, engine='pyarrow')

    return data


# Словарь, сопоставляющий имена моделей с функциями
model_functions = {
    "MNB_BoW_NB2": MNB_BoW_NB2,
    "Logreg_TFIDF": Logreg_TFIDF,
    "SGDClassifier": SGDClassifier
}

# Словарь для хранения данных обученных моделей
trained_models_info = {}

# Эндпоинт для обучения активной модели
@app.post("/train_model", response_model=TrainModelResponse)
async def train_model(
    request: str = Form('{"hyperparameters": {}}',
    description="Dict of hyperparameters as JSON string"),
    train_file: UploadFile = File(..., description="Training dataset in pq format"),
    test_file: UploadFile = File(..., description="Testing dataset in pq format"),
):
    global active_model_id

    # Проверка ID модели
    if active_model_id == 'Logreg_TFIDF_NB2':
        logger.warning("Попытка обучить Logreg_TFIDF_NB2, заблокировано.")
        raise HTTPException(
            status_code=404,
            detail="""Обучение Logreg_TFIDF_NB2 занимает слишком много времени,
пожалуйста, активируйте другую модель из списка"""
        )
    if active_model_id not in model_functions:
        logger.error('Модель с id "%s" не найдена для обучения.',
                     active_model_id)
        raise HTTPException(status_code=404, detail="Model not found")

    # Чтение данных из файлов
    train_data = await read_parquet_file(train_file)
    test_data = await read_parquet_file(test_file)

    X_train, y_train = train_data['text'], train_data['author']
    X_test, y_test = test_data['text'], test_data['author']

    # Получение функции обучения модели
    train_function = model_functions.get(active_model_id)

    try:
        request_data = json.loads(request)
        hyperparameters = request_data.get('hyperparameters', {})
    except json.JSONDecodeError as e:
        logger.error("Ошибка декодирования JSON для гиперпараметров: %s", e)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON format in 'request': {e}") from e

    start_time = time.perf_counter()
    try:
        metrics, model_path, learning_curve_data = train_function(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            **hyperparameters
        )
    except Exception as e:
        logger.error('Ошибка при обучении модели: %s', e)
        raise HTTPException(
            status_code=500,
            detail=f'Error training model: {e}') from e

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    execution_time = str(round(execution_time, 2))
    logger.info('Модель %s успешно обучена за %s секунд.',
                active_model_id, execution_time)

    # Генерация нового имени для модели
    base_name = active_model_id
    model_number = 1
    pattern = re.compile(f"{base_name}_(\\d+)")

    for key in initial_models_list.keys():
        match = pattern.match(key)
        if match:
            current_number = int(match.group(1))
            if current_number >= model_number:
                model_number = current_number + 1

    new_model_id = f"{base_name}_{model_number}"

    # Добавляем новую модель в основной словарь
    initial_models_list[new_model_id] = {
        'model': joblib.load(model_path),
        'description': 'Новая обученная модель'
    }

    # Добавляем информацию об обученной модели в словарь
    trained_models_info[new_model_id] = {
        "metrics": metrics,
        "execution_time": execution_time,
        "learning_curve_data": learning_curve_data
    }

    active_model_id = new_model_id

    response = TrainModelResponse(
        mod_id=new_model_id,
        execution_time=f"{execution_time} seconds",
        accuracy=str(metrics['accuracy']),
        precision=str(metrics['precision']),
        recall=str(metrics['recall']),
        f1=str(metrics['f1']),
        train_sizes=learning_curve_data['train_sizes'],
        train_scores_mean=learning_curve_data['train_scores_mean'],
        test_scores_mean=learning_curve_data['test_scores_mean']
    )

    return response


# Эндпоинт для сравнения экспериментов
@app.get("/experiments", status_code=HTTPStatus.OK)
async def get_experiments(exp_id: Union[str, None] = None) -> \
Union[Dict, list]:
    """Возвращает список всех обученных моделей или
данные одной модели."""
    if exp_id:
        if exp_id not in trained_models_info:
            logger.warning("Эксперимент с id '%s' не найден.", exp_id)
            raise HTTPException(status_code=404,
                                detail="Experiment not found")
        logger.info("Получение данных для эксперимента с id '%s'.",
                    exp_id)
        return trained_models_info[exp_id]["learning_curve_data"]
    else:
        experiments = [{"id": model_id, "name": f"Модель {model_id}"} \
                       for model_id in trained_models_info.keys()]
        logger.info("Возвращается список всех обученных моделей.\
Количество моделей: %d.", len(experiments))
        return experiments


# Эндпоинт для дообучения модели SGD
@app.post("/partial_fit", response_model=Dict[str, str],
          status_code=HTTPStatus.OK)
async def partial_fit(request_file: UploadFile = File()) -> Dict[str, str]:
    """
    Частично дообучает модель SGD с использованием новых данных.
    Параметры:
        request_file: Файл, содержащий новые обучающие данные.
    Возвращает:
        Сообщение, подтверждающее успешное обновление модели.
    """
    global active_model_id
    mod_id = "SGDClassifier"
    pipeline = initial_models_list[mod_id]['model']

    # Считываем данные для обучения
    train_data = await read_parquet_file(request_file)
    X_train, y = train_data['text'], train_data['author']

    # Извлекаем TfidfVectorizer и SGDClassifier из пайплайна
    vectorizer = pipeline.named_steps['tfidfvectorizer']
    model = pipeline.named_steps['sgdclassifier']

    # Преобразуем текстовые данные в tf-idf представление
    X_tfidf = vectorizer.transform(X_train)

    # Загрузка классов из файла
    loaded_classes = joblib.load('models/authors.pkl')

    # Обучаем модель на новых данных
    try:
        model.partial_fit(X_tfidf, y, classes=loaded_classes)
    except Exception as e:
        logger.error('Ошибка при дообучении модели: %s', e)
        raise HTTPException(
            status_code=500,
            detail=f'Error during partial fitting. Details: {e}') from e

    # Обновляем модель в пайплайне
    pipeline.named_steps['sgdclassifier'] = model

    new_model_id = 'SGDClassifier_2'
    joblib.dump(pipeline, f'models/{new_model_id}.joblib')

    # Добавляем новую модель в основной словарь
    initial_models_list[new_model_id] = {
        'model': joblib.load(f'models/{new_model_id}.joblib'),
        'description': 'Новая дообученная модель'
    }

    active_model_id = new_model_id

    logger.info("Модель с id '%s' успешно дообучена с новыми данными.",
                new_model_id)

    return {"message": f"Model with id '{new_model_id}' \
successfully updated with new data."}


# Запуск приложения
if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
