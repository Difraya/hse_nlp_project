from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
import joblib

def model2(X_train, y_train, X_test, y_test, model_path='ngram_naive_bayes_2.joblib', **hparams):
    # Определение и обучение модели
    ngram_naive_bayes = make_pipeline(
        CountVectorizer(ngram_range=(1, 2), max_features=500),
        MultinomialNB()
    )

    ngram_naive_bayes.fit(X_train, y_train)

    # Предсказание
    ngram_naive_bayes_preds = ngram_naive_bayes.predict(X_test)

    # Вычисление метрик
    metrics = {
        'accuracy': accuracy_score(y_test, ngram_naive_bayes_preds),
        'precision': precision_score(y_test, ngram_naive_bayes_preds, average='macro'),
        'recall': recall_score(y_test, ngram_naive_bayes_preds, average='macro'),
        'f1': f1_score(y_test, ngram_naive_bayes_preds, average='macro')
    }

    # Вывод метрик
    print(f'''Модель ngram_naive_bayes обучена на предоставленных данных
и сохранена в ngram_naive_bayes_2.joblib.
Получены метрики:
- Точность (Accuracy): {metrics['accuracy']}
- Точность (Precision): {metrics['precision']}
- Полнота (Recall): {metrics['recall']}
- F1-мера: {metrics['f1']}''')

    # Сохранение модели
    joblib.dump(ngram_naive_bayes, model_path)

    return metrics
