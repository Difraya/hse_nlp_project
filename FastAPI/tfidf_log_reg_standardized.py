from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def model3(X_train, y_train, X_test, y_test, model_path='tfidf_log_reg_standardized_2.joblib'):
    # Определение и обучение модели
    tfidf_log_reg_standardized = make_pipeline(
        TfidfVectorizer(max_features=333),
        StandardScaler(with_mean=False),
        LogisticRegression(random_state=42, max_iter=10000, solver='lbfgs')
    )
    
    tfidf_log_reg_standardized.fit(X_train, y_train)
    
    # Предсказание и вычисление точности
    tfidf_preds_standardized = tfidf_log_reg_standardized.predict(X_test)
    tfidf_acc_standardized = accuracy_score(y_test, tfidf_preds_standardized)
    print(f'''Модель tfidf_preds_standardized обучена на предоставленных данных
и сохранена в tfidf_log_reg_standardized_2.joblib.
Получена точность: {tfidf_acc_standardized}''')

    # Сохранение модели
    joblib.dump(tfidf_log_reg_standardized, model_path)

    return tfidf_acc_standardized