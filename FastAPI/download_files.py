import subprocess
from tqdm import tqdm

# Список команд для загрузки файлов с Kaggle
commands = [
    # "kaggle datasets download vorvit/books-eng --file df_test.pq --force",
    # "kaggle datasets download vorvit/books-eng --file df_train.pq --force",
    "kaggle datasets download vorvit/books-eng --file pipeline.joblib --force",
    "kaggle datasets download vorvit/books-eng --file train.pq --force",
    "kaggle datasets download vorvit/books-eng --file test.pq --force",
    # "kaggle datasets download vorvit/books-eng --file filtered_df.pq --force",
]


# Функция, которая выполняет команду и выводит результат
def run_command(command):
    """
    Run a shell command using subprocess and print its output.
    Args:
        command (str): The command to be executed.
    Returns:
        None
    """
    try:
        # subprocess.run выполняет команду и возвращает результат
        result = subprocess.run(command, check=True, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Output:")
        print(result.stdout.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error message: {e.stderr.decode('utf-8')}")


# Проход по списку команд и их выполнение с индикацией загрузки
for cmd in tqdm(commands, desc="Downloading files", unit="file"):
    run_command(cmd)
