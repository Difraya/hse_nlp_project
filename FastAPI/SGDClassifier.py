from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib


def model4(X_train, y_train, X_test, y_test,
           model_path='SGDClassifier_2.joblib', **hparams):
    """
    Trains an SGDClassifier with a pipeline including TfidfVectorizer and StandardScaler,
    evaluates it on the test set, and saves the trained model.

    Parameters:
    - X_train: list or array-like
        The training input samples.
    - y_train: list or array-like
        The target values (class labels) for training.
    - X_test: list or array-like
        The testing input samples.
    - y_test: list or array-like
        The true target values (class labels) for testing.
    - model_path: str, optional
        The path where the trained model is saved. Default is 'SGDClassifier_2.joblib'.
    - hparams: dict, optional
        Hyperparameters for the SGDClassifier, such as 'alpha' and 'max_iter'.

    Returns:
    - metrics: dict
        A dictionary containing accuracy, precision, recall, and F1 score of the model.
    """
    # Определяем гиперпараметры по умолчанию
    default_hparams = {
        'alpha': 0.001,
        'max_iter': 10000,
    }

    # Если hparams не задан или пуст, используем параметры по умолчанию
    if not hparams:
        hparams = default_hparams
    else:
        # Объединяем hparams с default_hparams, обновляя значения по умолчанию
        for key, value in default_hparams.items():
            hparams.setdefault(key, value)

    # Определение и обучение модели
    SGDpipe = make_pipeline(
        TfidfVectorizer(max_features=700),
        StandardScaler(with_mean=False),
        SGDClassifier(
            alpha=hparams.get('alpha', 0.001),
            max_iter=hparams.get('max_iter', 10000)
        )
    )

    SGDpipe.fit(X_train, y_train)

    # Предсказание
    SGDpipe_preds = SGDpipe.predict(X_test)

    # Вычисление метрик
    metrics = {
        'accuracy': accuracy_score(y_test, SGDpipe_preds),
        'precision': precision_score(y_test, SGDpipe_preds, average='macro'),
        'recall': recall_score(y_test, SGDpipe_preds, average='macro'),
        'f1': f1_score(y_test, SGDpipe_preds, average='macro')
    }

    # Вывод метрик
    print(f'''Модель SGDClassifier обучена на предоставленных данных
и сохранена в SGDClassifier_2.joblib.
Получены метрики:
- Точность (Accuracy): {metrics['accuracy']}
- Точность (Precision): {metrics['precision']}
- Полнота (Recall): {metrics['recall']}
- F1-мера: {metrics['f1']}''')

    # Сохранение модели
    joblib.dump(SGDpipe, model_path)

    return metrics
