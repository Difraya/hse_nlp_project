import subprocess

#Список команд для загрузки файлов с Kaggle
commands = [
    "kaggle datasets download vorvit/books-eng --file pipeline.joblib --force",
    "kaggle datasets download vorvit/books-eng --file ngram_naive_bayes.joblib --force",
    "kaggle datasets download vorvit/books-eng --file tfidf_log_reg_standardized.joblib --force",
    "kaggle datasets download vorvit/books-eng --file SGDClassifier.joblib --force",
    "kaggle datasets download vorvit/books-eng --file df_train.pq --force",
    "kaggle datasets download vorvit/books-eng --file df_test.pq --force"
]

#Функция, которая выполняет команду и выводит результат
def run_command(command):
    try:
        # subprocess.run выполняет команду и возвращает результат
        result = subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Output:\n{result.stdout.decode('utf-8')}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}\nError message: {e.stderr.decode('utf-8')}")

#Проход по списку команд и их выполнение
for cmd in commands:
    run_command(cmd)