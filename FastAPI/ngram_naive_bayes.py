from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import learning_curve
import joblib
import numpy as np


def model2(X_train, y_train, X_test, y_test,
           model_path='ngram_naive_bayes_2.joblib', **hparams):
    """
    Train a Naive Bayes model using n-grams and evaluate it on the test set.
    Parameters:
    - X_train: Training data
    - y_train: Training labels
    - X_test: Test data
    - y_test: Test labels
    - model_path: Path to save the trained model
    - hparams: Additional hyperparameters
    Returns:
    - metrics: Dictionary containing evaluation metrics
    """

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
        'precision': precision_score(y_test,
                                     ngram_naive_bayes_preds, average='macro'),
        'recall': recall_score(y_test,
                               ngram_naive_bayes_preds, average='macro'),
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

    # Построение кривой обучения
    train_sizes, train_scores, test_scores = learning_curve(
        ngram_naive_bayes,
        X_train,
        y_train,
        cv=2,
        scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5)
    )

    # Среднее и стандартное отклонение для кривой обучения
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    learning_curve_data = {
        'train_sizes': train_sizes.tolist(),
        'train_scores_mean': train_scores_mean.tolist(),
        'test_scores_mean': test_scores_mean.tolist()
    }

    # Вывод данныx кривой обучения
    print(f'''Получены данные кривой обучения:
- train_sizes: {learning_curve_data['train_sizes']},
- train_scores_mean: {learning_curve_data['train_scores_mean']},
- test_scores_mean: {learning_curve_data['test_scores_mean']}''')

    # Сохранение модели
    joblib.dump(ngram_naive_bayes, model_path)

    return (metrics, model_path, learning_curve_data)
