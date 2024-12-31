from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def model3(X_train, y_train, X_test, y_test, model_path='tfidf_log_reg_standardized_2.joblib', **hparams):
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
    print(f'''Модель tfidf_log_reg_standardized обучена на предоставленных данных
и сохранена в tfidf_log_reg_standardized_2.joblib.
Получены метрики:
- Точность (Accuracy): {metrics['accuracy']}
- Точность (Precision): {metrics['precision']}
- Полнота (Recall): {metrics['recall']}
- F1-мера: {metrics['f1']}''')

    # Сохранение модели
    joblib.dump(tfidf_log_reg_standardized, model_path)

    return metrics