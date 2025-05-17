import asyncio
import logging
from pathlib import Path
import json
import joblib
import base64
import pandas as pd
import aiohttp
from aiogram import Bot, Dispatcher, types, Router, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.fsm.storage.memory import MemoryStorage
from writers import writers_dict
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Настройка логирования
logging.basicConfig(
    filename='logs/bot_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# Вывод в консоль
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    fmt='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Конфигурация
API_KEYS = {
    'FUSION_BRAIN': {
        'API_KEY': '',
        'SECRET_KEY': ''
    },
    'TELEGRAM': ''
}

MODEL_PATH = Path(__file__).resolve().parent.parent / 'FastAPI' / 'models' / 'pipeline.joblib'
IMAGES_PATH = Path('images')
GPT_MODEL_PATH = "C:/_1/2/author_style_gpt2"  # "E:/__/_1/1/gpt250"
GPT_TOK_PATH = "E:/__/_1/1/cus_tok"
AUTHORS_DF = pd.read_parquet('limited.pq')
UNIQUE_AUTHORS = AUTHORS_DF['author'].unique().tolist()
WRITERS_DICT = {author: author for author in UNIQUE_AUTHORS}

# Инициализация объектов
bot = Bot(token=API_KEYS['TELEGRAM'])
dp = Dispatcher(storage=MemoryStorage())
router = Router()

# Загрузка моделей
model = joblib.load(MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка GPT-модели для генерации текста
gpt_tokenizer = GPT2Tokenizer.from_pretrained(GPT_MODEL_PATH)
gpt_model = GPT2LMHeadModel.from_pretrained(GPT_MODEL_PATH).to(device)

# Изменения в классе состояний
class Form(StatesGroup):
    choosing_function = State()
    waiting_for_author = State()
    waiting_for_text = State()
    waiting_for_prediction = State()
    waiting_for_image_text = State()
    processing = State()

# Клавиатура
FUNCTION_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Предсказание авторства")],
        [KeyboardButton(text="Текст в стиле автора")],
        [KeyboardButton(text="Генерация изображения")],
        [KeyboardButton(text="Вернуться в меню")]
    ],
    resize_keyboard=True
)

# Глобальная переменная для отслеживания активных пользователей
active_users = set()

# Инициализация API
class AsyncFusionBrainAPI:
    def __init__(self):
        self.base_url = 'https://api-key.fusionbrain.ai/key/api/v1/'
        self.headers = {
            'X-Key': f'Key {API_KEYS["FUSION_BRAIN"]["API_KEY"]}',
            'X-Secret': f'Secret {API_KEYS["FUSION_BRAIN"]["SECRET_KEY"]}',
        }
        self.session = aiohttp.ClientSession()

    async def get_pipeline(self):
        async with self.session.get(f'{self.base_url}pipelines', headers=self.headers) as response:
            data = await response.json()
            return data[0]['id']

    async def generate_image(self, prompt, pipeline_id):
        params = {
            "type": "GENERATE",
            "numImages": 1,
            "width": 768,
            "height": 768,
            "generateParams": {"query": prompt}
        }
        data = aiohttp.FormData()
        data.add_field('pipeline_id', pipeline_id)
        data.add_field('params', json.dumps(params), content_type='application/json')
        async with self.session.post(f'{self.base_url}pipeline/run', headers=self.headers, data=data) as response:
            result = await response.json()
            return result['uuid']

    async def check_generation(self, uuid):
        for _ in range(20):
            async with self.session.get(f'{self.base_url}pipeline/status/{uuid}', headers=self.headers) as response:
                data = await response.json()
                if data['status'] == 'DONE':
                    return data['result']['files']
                await asyncio.sleep(2)
        return []

    async def close(self):
        await self.session.close()


async def load_author_images():
    """Предзагрузка изображений авторов в память"""
    global author_image_cache
    author_image_cache = {}
    for author in writers_dict:
        image_path = IMAGES_PATH / f'{author}.jpg'
        if image_path.exists():
            with open(image_path, 'rb') as f:
                author_image_cache[author] = f.read()


def sanitize_prompt(prompt: str) -> str:
    """Очищает промпт для использования в API"""
    return (
        prompt.replace("\n", " ")
        .replace("[AUTHOR_", "")
        .replace("]", "")
        .replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .strip()[:200]
    )


async def predict_authors(text):
    """Асинхронное предсказание авторов"""
    text_series = pd.Series([text])
    probas = model.predict_proba(text_series)[0]
    return dict(zip(model.classes_, probas))


async def generate_text(author, prompt, max_new_tokens=150):
    """Генерация текста в стиле автора"""
    input_text = f"[AUTHOR_{author}] {prompt}"
    
    # Токенизация
    encoded_inputs = gpt_tokenizer(
        input_text,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)
    
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']
    
    # Генерация
    outputs = gpt_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.2,
        pad_token_id=gpt_tokenizer.eos_token_id,
        num_return_sequences=1,
        do_sample=True
    )
    
    # Декодирование
    generated_text = gpt_tokenizer.decode(outputs[0], skip_special_tokens=False)
    return generated_text.replace(f'[AUTHOR_{author}]', '').strip()


async def process_message(message: types.Message, fb_api: AsyncFusionBrainAPI, generate_image: bool = False, run_prediction: bool = True):
    try:
        active_users.add(message.from_user.id)
        text = sanitize_prompt(message.text[:4096])
        
        processing_msg = await bot.send_message(message.chat.id, "⏳ Обрабатываю запрос...")

        predictions = None
        image_data = None
        response_text = []
        uuid = None

        tasks = {}
        if run_prediction:
            tasks['prediction'] = asyncio.create_task(predict_authors(text))
        if generate_image:
            pipeline_id = await fb_api.get_pipeline()
            tasks['generation'] = asyncio.create_task(fb_api.generate_image(text, pipeline_id))

        if tasks:
            done, _ = await asyncio.wait(tasks.values())
            
            for task in done:
                if task == tasks.get('prediction'):
                    predictions = task.result()
                elif task == tasks.get('generation'):
                    uuid = task.result()

        # Генерация изображения
        if generate_image and uuid:
            try:
                if files := await fb_api.check_generation(uuid):
                    file_url = files[0]
                    data = file_url.split(',', 1)[1] if file_url.startswith('data:image') else file_url
                    image_data = base64.b64decode(data)
            except Exception as e:
                logger.error(f"Image generation failed: {str(e)}")
                await processing_msg.edit_text("⚠️ Не удалось сгенерировать изображение")

        # Формирование текстового ответа
        if run_prediction and predictions:
            sorted_authors = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
            response_text.append("📚 Результаты анализа:")
            for i, (author, prob) in enumerate(sorted_authors):
                writer_name = writers_dict.get(author, "Неизвестный автор")
                response_text.append(f"{'🥇' if i ==0 else '🥈' if i==1 else '🥉'} {writer_name}: {prob*100:.2f}%")

        # Отправка результата
        final_message = '\n'.join(response_text) if response_text else "✅ Готово!"
        
        if image_data:
            await processing_msg.delete()
            await message.reply_photo(
                photo=BufferedInputFile(image_data, "image.jpg"),
                caption=final_message
            )
        else:
            await processing_msg.edit_text(final_message)

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        await bot.send_message(message.chat.id, "❌ Произошла ошибка при обработке")

    finally:
        active_users.discard(message.from_user.id)


@router.message(Command('start'))
async def start_command(message: types.Message, state: FSMContext):
    await state.set_state(Form.choosing_function)
    description = (
        "Добро пожаловать в бота для анализа текста и генерации текста в стиле известных авторов!\n\n"
        "Функционал:\n"
        "1. Предсказание авторства — определяет автора текста\n"
        "2. Текст в стиле автора — продолжает текст в стиле выбранного автора\n"
        "3. Генерация изображения — создаёт изображение на основе текста\n\n"
        "Выберите одну из функций ниже:"
    )
    await message.reply(description, reply_markup=FUNCTION_KEYBOARD)


@router.message(Command('help'))
async def help_command(message: types.Message):
    description = (
        "Добро пожаловать в бота для анализа текста и генерации текста в стиле известных авторов!\n\n"
        "Функционал:\n"
        "1. Предсказание авторства — определяет автора текста\n"
        "2. Текст в стиле автора — продолжает текст в стиле выбранного автора\n"
        "3. Генерация изображения — создаёт изображение на основе текста\n\n"
        "Чтобы начать, нажмите на одну из кнопок в меню"
    )
    await message.reply(description, reply_markup=FUNCTION_KEYBOARD)


@router.message(Form.choosing_function, F.text == "Вернуться в меню")
@router.message(Form.waiting_for_prediction, F.text == "Вернуться в меню")
@router.message(Form.waiting_for_author, F.text == "Вернуться в меню")
@router.message(Form.waiting_for_text, F.text == "Вернуться в меню")
async def back_to_menu(message: types.Message, state: FSMContext):
    """Обработка кнопки возврата в меню"""
    await state.set_state(Form.choosing_function)
    description = (
        "Добро пожаловать в бота для анализа текста и генерации текста в стиле известных авторов!\n\n"
        "Функционал:\n"
        "1. Предсказание авторства — определяет автора текста\n"
        "2. Текст в стиле автора — продолжает текст в стиле выбранного автора\n"
        "3. Генерация изображения — создаёт изображение на основе текста\n\n"
        "Выберите одну из функций ниже:"
    )
    await message.reply(description, reply_markup=FUNCTION_KEYBOARD)


@router.message(Form.choosing_function, F.text == "Предсказание авторства")
async def prediction_selected(message: types.Message, state: FSMContext):
    if message.from_user.id in active_users:
        await message.reply("Пожалуйста, дождитесь завершения предыдущей операции")
        return

    await state.set_state(Form.waiting_for_prediction)
    await message.reply("Отправьте текст для анализа авторства")


@router.message(Form.choosing_function, F.text == "Текст в стиле автора")
async def generation_selected(message: types.Message, state: FSMContext):
    if message.from_user.id in active_users:
        await message.reply("Пожалуйста, дождитесь завершения предыдущей операции")
        return

    await state.set_state(Form.waiting_for_author)

    builder = ReplyKeyboardBuilder()
    for author in WRITERS_DICT.values():
        builder.add(KeyboardButton(text=writers_dict.get(author, "Неизвестный автор")))

    builder.adjust(2)

    await message.reply(
        "Выберите автора из списка:",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )


@router.message(Form.waiting_for_author, F.text == "Вернуться в меню")
async def back_to_menu_from_author_selection(message: types.Message, state: FSMContext):
    await state.set_state(Form.choosing_function)
    description = (
        "Добро пожаловать в бота для анализа текста и генерации текста в стиле известных авторов!\n"
        "Функционал:\n"
        "1. Предсказание авторства — определяет автора текста\n"
        "2. Текст в стиле автора — продолжает текст в стиле выбранного автора\n"
        "3. Генерация изображения — создаёт изображение на основе текста\n"
        "Выберите одну из функций ниже:"
    )
    await message.reply(description, reply_markup=FUNCTION_KEYBOARD)


# Обработчик для генерации изображения
@router.message(Form.choosing_function, F.text == "Генерация изображения")
async def image_generation_selected(message: types.Message, state: FSMContext):
    if message.from_user.id in active_users:
        await message.reply("Пожалуйста, дождитесь завершения предыдущей операции")
        return
    await state.set_state(Form.waiting_for_image_text)
    await message.reply("Отправьте текст для генерации изображения")


# Обработчик для генерации изображения по тексту
@router.message(Form.waiting_for_image_text)
async def handle_image_generation(message: types.Message, state: FSMContext):
    if message.from_user.id in active_users:
        return await message.reply("⏳ Пожалуйста, дождитесь завершения предыдущей операции")

    processing_msg = await bot.send_message(message.chat.id, "🎨 Генерирую изображение...")
    
    fb_api = AsyncFusionBrainAPI()
    try:
        text = sanitize_prompt(message.text[:200])
        pipeline_id = await fb_api.get_pipeline()
        uuid = await fb_api.generate_image(text, pipeline_id)
        
        if files := await fb_api.check_generation(uuid):
            file_url = files[0]
            data = file_url.split(',', 1)[1] if file_url.startswith('data:image') else file_url
            image_data = base64.b64decode(data)
            
            await processing_msg.delete()
            await message.reply_photo(
                photo=BufferedInputFile(image_data, "generated_image.jpg"),
                caption=f"Изображение по тексту: {text[:50]}..."
            )
        else:
            await processing_msg.edit_text("⚠️ Не удалось сгенерировать изображение")

    except Exception as e:
        logger.error(f"Ошибка генерации изображения: {str(e)}")
        await processing_msg.edit_text("❌ Ошибка при генерации изображения")
    
    finally:
        await fb_api.close()
        await state.set_state(Form.choosing_function)


# Обработчик предсказания авторства
@router.message(Form.waiting_for_prediction)
async def handle_prediction_text(message: types.Message, state: FSMContext):
    if message.from_user.id in active_users:
        await message.reply("Пожалуйста, дождитесь завершения предыдущей операции")
        return

    fb_api = AsyncFusionBrainAPI()
    try:
        processing_msg = await bot.send_message(message.chat.id, "⏳ Анализирую текст...")
        
        # Получаем предсказания
        text = sanitize_prompt(message.text)
        predictions = await predict_authors(text)
        
        # Сортируем результаты
        sorted_authors = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
        top_author = sorted_authors[0][0] if sorted_authors else None
        
        # Формируем сообщение
        response_text = ["📚 Результаты анализа авторства:"]
        for i, (author, prob) in enumerate(sorted_authors):
            writer_name = writers_dict.get(author, "Неизвестный автор")
            response_text.append(f"{'🥇' if i ==0 else '🥈' if i==1 else '🥉'} {writer_name}: {prob*100:.2f}%")
        
        # Отправляем фото автора
        if top_author and top_author in author_image_cache:
            await bot.send_photo(
                chat_id=message.chat.id,
                photo=BufferedInputFile(
                    author_image_cache[top_author],
                    filename=f"{top_author}.jpg"
                ),
                caption=f"Скорее всего автор: {writers_dict[top_author]}"
            )
        
        # Отправляем текстовый ответ
        await processing_msg.edit_text('\n'.join(response_text))

    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        await message.reply("❌ Произошла ошибка при анализе текста")
    
    finally:
        await fb_api.close()
        await state.set_state(Form.choosing_function)


@router.message(Form.waiting_for_author)
async def handle_author_selection(message: types.Message, state: FSMContext):
    selected_author = None
    for eng, rus in writers_dict.items():
        if rus == message.text:
            selected_author = eng
            break
    if not selected_author:
        logger.error(f"Автор {message.text} не найден")
        await message.reply("Неизвестный автор. Пожалуйста, выберите из списка:")
        return
    await state.update_data(selected_author=selected_author)
    await state.set_state(Form.waiting_for_text)
    await message.reply("Введите текст для продолжения в стиле выбранного автора", reply_markup=ReplyKeyboardRemove())


# Обработчик генерации текста
@router.message(Form.waiting_for_text)
async def handle_text_generation(message: types.Message, state: FSMContext):
    if message.from_user.id in active_users:
        return await message.reply("⏳ Пожалуйста, дождитесь предыдущей операции")

    active_users.add(message.from_user.id)
    processing_msg = None

    try:
        data = await state.get_data()
        author = data['selected_author']
        prompt = message.text[:4096]

        processing_msg = await bot.send_message(
            message.chat.id,
            "🧠 Генерирую текст и изображение..."
        )

        generated_text = await generate_text(author, prompt)
        
        # Генерация изображения
        image_prompt = sanitize_prompt(generated_text[:200])
        fb_api = AsyncFusionBrainAPI()
        try:
            pipeline_id = await fb_api.get_pipeline()
            uuid = await fb_api.generate_image(image_prompt, pipeline_id)
            
            if files := await fb_api.check_generation(uuid):
                file_url = files[0]
                data_part = file_url.split(',', 1)[1] if file_url.startswith('data:image') else file_url
                image_data = base64.b64decode(data_part)
                
                await message.reply_photo(
                    photo=BufferedInputFile(image_data, "generated_image.jpg"),
                    caption=f"✍️ Текст в стиле {writers_dict[author]}:\n\n{generated_text[:800]}"
                )
            else:
                await message.reply(f"✍️ Текст в стиле {writers_dict[author]}:\n\n{generated_text}")

        except Exception as e:
            logger.error(f"Ошибка генерации изображения: {str(e)}")
            await message.reply(f"⚠️ Не удалось сгенерировать изображение, но вот ваш текст:\n\n{generated_text}")

    except Exception as e:
        logger.error(f"Ошибка генерации: {str(e)}")
        await message.reply("❌ Произошла ошибка при генерации")

    finally:
        if processing_msg:
            await processing_msg.delete()
        
        # Возвращаем в главное меню
        await state.set_state(Form.choosing_function)
        await message.answer(
            "Выберите следующее действие:",
            reply_markup=FUNCTION_KEYBOARD
        )
        active_users.discard(message.from_user.id)

async def main():
    await load_author_images()
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())