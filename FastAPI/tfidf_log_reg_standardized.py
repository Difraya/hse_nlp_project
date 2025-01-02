from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def model3(X_train, y_train, X_test, y_test,
           model_path='tfidf_log_reg_standardized_2.joblib', **hparams):
    """
    Trains a logistic regression model with a pipeline that includes TfidfVectorizer and StandardScaler,
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
        The path where the trained model is saved. Default is 'tfidf_log_reg_standardized_2.joblib'.
    - hparams: dict, optional
        Hyperparameters for the LogisticRegression, such as 'random_state', 'max_iter', and 'solver'.

    Returns:
    - metrics: dict
        A dictionary containing accuracy, precision, recall, and F1 score of the model.
    """
    # Определяем гиперпараметры по умолчанию
    default_hparams = {
        'random_state': 42,
        'max_iter': 10000,
        'solver': 'lbfgs'
    }

    # Если hparams не задан или пуст, используем параметры по умолчанию
    if not hparams:
        hparams = default_hparams
    else:
        # Объединяем hparams с default_hparams, обновляя значения по умолчанию
        for key, value in default_hparams.items():
            hparams.setdefault(key, value)

    # Определение и обучение модели
    tfidf_log_reg_standardized = make_pipeline(
        TfidfVectorizer(max_features=333),
        StandardScaler(with_mean=False),
        LogisticRegression(
            random_state=hparams.get('random_state', 42),
            max_iter=hparams.get('max_iter', 10000),
            solver=hparams.get('solver', 'lbfgs')
            )
    )

    tfidf_log_reg_standardized.fit(X_train, y_train)

    # Предсказание
    tfidf_preds = tfidf_log_reg_standardized.predict(X_test)

    # Вычисление метрик
    metrics = {
        'accuracy': accuracy_score(y_test, tfidf_preds),
        'precision': precision_score(y_test, tfidf_preds, average='macro'),
        'recall': recall_score(y_test, tfidf_preds, average='macro'),
        'f1': f1_score(y_test, tfidf_preds, average='macro')
    }

    # Вывод метрик
    print(f'''Модель tfidf_log_reg_standardized обучена на
предоставленных данных и сохранена в tfidf_log_reg_standardized_2.joblib.
Получены метрики:
- Точность (Accuracy): {metrics['accuracy']}
- Точность (Precision): {metrics['precision']}
- Полнота (Recall): {metrics['recall']}
- F1-мера: {metrics['f1']}''')

    # Сохранение модели
    joblib.dump(tfidf_log_reg_standardized, model_path)

    return (metrics, model_path)
