import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

def model4(hparams, X_train, y_train, X_test, y_test, model_path='SGDClassifier_2.joblib'):
    # Определяем гиперпараметры по умолчанию
    default_hparams = {
        'loss': 'hinge',
        'penalty': 'l2',
        'alpha': 0.01,
        'random_state': 42,
        'max_iter': 10000,

    }

    # Если hparams не задан, используем параметры по умолчанию
    if hparams is None or {}:
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
                  loss=hparams.get('loss', 'hinge'),
                  penalty=hparams.get('penalty', 'l2'),
                  alpha=hparams.get('alpha', 0.0001),
                  random_state=hparams.get('random_state'),

                  )
    )

    SGDpipe.fit(X_train, y_train)

    # Предсказание и вычисление точности
    SGDpipe_preds = SGDpipe.predict(X_test)
    SGDpipe_acc = accuracy_score(y_test, SGDpipe_preds)
    print(f'''Модель SGDClassifier обучена на предоставленных данных
и сохранена в SGDClassifier_2.joblib.
Получена точность: {SGDpipe_acc}''')

    # Сохранение модели
    joblib.dump(SGDpipe, model_path)

    return SGDpipe_acc
