import logging
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments,
    TrainerCallback, EarlyStoppingCallback
)
from torch.utils.data import Dataset
from datetime import datetime

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Кастомный callback для логирования
class CustomLoggingCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        logger.info(f"\n{'='*30} Начало эпохи {state.epoch}/{args.num_train_epochs} {'='*30}")
        logger.info(f"Общее количество шагов: {state.max_steps}")
        logger.info(f"Текущий learning rate: {self._get_current_lr(kwargs['optimizer']):.2e}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            current_step = state.global_step
            logger.info(
                f"Шаг {current_step}/{state.max_steps} | "
                f"Loss: {logs.get('loss', 'недоступно')} | "
                f"Learning Rate: {self._get_current_lr(kwargs['optimizer']):.2e} | "
                f"Время: {datetime.now().strftime('%H:%M:%S')}"
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(f"\n{'='*30} Конец эпохи {state.epoch}/{args.num_train_epochs} {'='*30}")
        epoch_logs = [log for log in state.log_history if log.get('epoch') == state.epoch]
        train_loss = next(
            (log['loss'] for log in epoch_logs if 'loss' in log),
            'недоступно'
        )
        loss_str = f"{train_loss:.4f}" if isinstance(train_loss, float) else str(train_loss)
        logger.info(f"Средний лосс за эпоху: {loss_str}")
        logger.info(f"Общее время обучения: {state.total_flos / 1e9:.2f} секунд")

    def _get_current_lr(self, optimizer):
        return optimizer.param_groups[0]['lr']

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Используемое устройство: {device}")

# 1. Загрузка и предобработка данных
df = pd.read_parquet('limited.pq')

# Оставить случайные строк на каждого автора
df = df.groupby('author', group_keys=False).sample(n=33, random_state=42).reset_index(drop=True)

df['processed_text'] = df.apply(lambda x: f"[AUTHOR_{x['author']}] {x['text']}", axis=1)

# Разделение данных на train/validation
train_texts, val_texts = train_test_split(
    df['processed_text'].tolist(),
    test_size=0.1, 
    random_state=42
)

# 2. Инициализация токенизатора
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

authors = df['author'].unique().tolist()
new_tokens = [f'[AUTHOR_{author}]' for author in authors]
tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
tokenizer.pad_token = tokenizer.eos_token

# 3. Кастомный датасет
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

# Создание датасетов
train_dataset = TextDataset(train_texts, tokenizer)
eval_dataset = TextDataset(val_texts, tokenizer)

# 4. Инициализация модели
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model.to(device)

# 5. Настройка обучения
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    fp16=True,
    dataloader_pin_memory=True,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_steps=100,
    weight_decay=0.01,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[
        CustomLoggingCallback(),
        EarlyStoppingCallback(early_stopping_patience=5)
    ],
)

# 6. Запуск обучения
logger.info("\n" + "="*50)
logger.info("Начало обучения модели")
logger.info(f"Обучающих примеров: {len(train_dataset)}")
logger.info(f"Валидационных примеров: {len(eval_dataset)}")
logger.info(f"Общее количество шагов: {training_args.max_steps}")
logger.info(f"Ранняя остановка после 3 эпох без улучшений")
logger.info("="*50 + "\n")

try:
    trainer.train()
except Exception as e:
    logger.error(f"Ошибка во время обучения: {str(e)}")
    raise

# 7. Сохранение лучшей модели
model.save_pretrained('./author_style_gpt2')
tokenizer.save_pretrained('./author_style_gpt2')

# 8. Функция генерации с проверкой автора
def generate_text(author, prompt, max_length=100):
    if author not in authors:
        raise ValueError(f"Автор {author} не найден. Доступные: {', '.join(authors)}")
    
    model = GPTNeoForCausalLM.from_pretrained('./author_style_gpt2').to(device)
    tokenizer = AutoTokenizer.from_pretrained('./author_style_gpt2')

    input_text = f"[AUTHOR_{author}] {prompt}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    clean_text = generated_text.replace(f'[AUTHOR_{author}]', '').strip()
    return clean_text

# Пример использования
print(generate_text('Fyodor_Dostoyevsky', 'The meaning of life is'))