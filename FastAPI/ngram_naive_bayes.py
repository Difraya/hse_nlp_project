import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

def model2(hparams, X_train, y_train, X_test, y_test, model_path='ngram_naive_bayes_2.joblib'):
    # Определение и обучение модели
    ngram_naive_bayes = make_pipeline(
        CountVectorizer(ngram_range=(1, 2), max_features=500),
        MultinomialNB(hparams)
    )

    ngram_naive_bayes.fit(X_train, y_train)

    # Предсказание и вычисление точности
    ngram_naive_bayes_preds = ngram_naive_bayes.predict(X_test)
    ngram_naive_bayes_acc = accuracy_score(y_test, ngram_naive_bayes_preds)
    print(f'''Модель ngram_naive_bayes обучена на предоставленных данных
и сохранена в ngram_naive_bayes_2.joblib.
Получена точность: {ngram_naive_bayes_acc}''')

    # Сохранение модели
    joblib.dump(ngram_naive_bayes, model_path)

    return ngram_naive_bayes_acc
