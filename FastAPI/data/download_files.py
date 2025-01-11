import subprocess
from tqdm import tqdm

# Определяем папки для сохранения файлов
models_folder = "models"
dataset_folder = "data"

# Пытаемся создать папки, и если они уже существуют, произойдет ошибка
subprocess.run(f"mkdir {models_folder}", shell=True)
subprocess.run(f"mkdir {dataset_folder}", shell=True)

# Список команд для загрузки файлов с Kaggle
commands = [
    f"kaggle datasets download vorvit/books-eng --file train.pq --force --path {dataset_folder}",
    f"kaggle datasets download vorvit/books-eng --file test.pq --force --path {dataset_folder}",
    f"kaggle datasets download vorvit/books-eng --file pipeline.joblib --force --path {models_folder}",
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