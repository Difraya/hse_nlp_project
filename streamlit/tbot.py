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
from googletrans import Translator
from langdetect import detect

# Настройка логирования
logging.basicConfig(
    filename='logs/bot_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# Добавляем вывод в консоль
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
        'API_KEY': 'CE5724FC57BC0BEE5421AEBEAB418E40',
        'SECRET_KEY': '93798A3536DE9355BE4C242EA715A0AF'
    },
    'TELEGRAM': '7563764834:AAHS0nX8CDH5_I5XA0IIAUmjQk_mU03MacA'
}

MODEL_PATH = Path(__file__).resolve().parent.parent / 'FastAPI' / 'models' / 'pipeline.joblib'
IMAGES_PATH = Path('images')
GPT_MODEL_PATH = "D:/__/_1/1/author_style_gpt2"  # "C:/_1/2/author_style_gpt2" "E:/__/_1/1/gpt250"
GPT_TOK_PATH = "D:/__/_1/1/cus_tok"
AUTHORS_DF = pd.read_parquet('limited.pq')
UNIQUE_AUTHORS = AUTHORS_DF['author'].unique().tolist()
WRITERS_DICT = {author: author for author in UNIQUE_AUTHORS}
JOKE_DF = pd.read_parquet('jokes.pq')

# Кэш показанных анекдотов для каждого пользователя
shown_jokes = {}  # Формат: {user_id: {author_rus: [показанные_анекдоты]}}

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

# Обновленная клавиатура
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

# Middleware для блокировки ввода
class ProcessingMiddleware:
    async def __call__(self, handler, event, data):
        state = data["state"]
        current_state = await state.get_state()
        if current_state == Form.processing:
            await event.message.reply("⏳ Бот занят. Подождите завершения текущей операции.")
            return
        return await handler(event, data)

dp.message.middleware(ProcessingMiddleware())

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
            "width": 640,
            "height": 640,
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


# Перевод на английский
translator = Translator()

async def translate_text(text: str, src_lang: str = 'ru', dest_lang: str = 'en') -> str:
    """Переводит текст с исходного языка на целевой (по умолчанию: ru → en)"""
    try:
        loop = asyncio.get_event_loop()
        translated = await loop.run_in_executor(None, translator.translate, text, dest_lang, src_lang)
        return translated.text
    except Exception as e:
        logger.error(f"Ошибка перевода: {str(e)}")
        return text


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
        "Добро пожаловать в бота для анализа и генерации текста в стиле известных авторов!\n\n"
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
    await message.reply("Введите текст на английском для анализа авторства")


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
        return await message.reply("⏳ Пожалуйста, дождитесь завершения предыдущей операции")
    await state.set_state(Form.waiting_for_image_text)
    await message.reply("Введите текст для генерации изображения")


# Обработчик для генерации изображения по тексту
@router.message(Form.waiting_for_image_text)
async def handle_image_generation(message: types.Message, state: FSMContext):
    if not message.text or message.text.strip() == "":  # Проверка на пустой текст
        return await message.reply("⚠️ Пожалуйста, введите текст для генерации изображения")

    if message.from_user.id in active_users:
        return await message.reply("⏳ Пожалуйста, дождитесь завершения предыдущей операции")

    active_users.add(message.from_user.id)
    await state.set_state(Form.processing)  # Установка состояния блокировки

    fb_api = AsyncFusionBrainAPI()
    try:
        processing_msg = await bot.send_message(message.chat.id, "🎨 Генерирую изображение...")
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
        await message.answer(
            "Выберите следующее действие:",
            reply_markup=FUNCTION_KEYBOARD
        )
        active_users.discard(message.from_user.id)


# Обработчик предсказания авторства с анекдотом
@router.message(Form.waiting_for_prediction)
async def handle_prediction_text(message: types.Message, state: FSMContext):
    if not message.text or message.text.strip() == "" or message.text in ['Предсказание авторства',
                                                                          'Текст в стиле автора',
                                                                          'Генерация изображения']:
        return await message.reply("⚠️ Пожалуйста, введите текст на английском для анализа авторства")

    if message.from_user.id in active_users:
        return await message.reply("⏳ Пожалуйста, дождитесь завершения предыдущей операции")

    active_users.add(message.from_user.id)
    await state.set_state(Form.processing)

    fb_api = AsyncFusionBrainAPI()
    try:
        processing_msg = await bot.send_message(message.chat.id, "⏳ Анализирую текст...")
        text = sanitize_prompt(message.text)
        translated_text = ''

        # Автоматический перевод текста на английский
        src_lang = detect(text)
        if src_lang != 'en':
            translated_text = await translate_text(text, src_lang=src_lang)
            if src_lang == 'ru':
                await message.reply(f"🔄 Текст переведён с русского на английский для анализа.")
                await message.reply(translated_text)

        if translated_text != '':
            predictions = await predict_authors(translated_text)
        else:
            predictions = await predict_authors(text)

        sorted_authors = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
        top_author_eng = sorted_authors[0][0] if sorted_authors else None

        # Получение русского имени автора
        top_author_rus = writers_dict.get(top_author_eng, "Неизвестный автор")

        # Формирование текстового ответа
        response_text = ["📚 Результаты анализа авторства:"]
        for i, (author_eng, prob) in enumerate(sorted_authors):
            writer_name = writers_dict.get(author_eng, "Неизвестный автор")
            response_text.append(f"{'🥇' if i ==0 else '🥈' if i==1 else '🥉'} {writer_name}: {prob*100:.2f}%")

        # Отправляем фото автора
        if top_author_eng and top_author_eng in author_image_cache:
            await bot.send_photo(
                chat_id=message.chat.id,
                photo=BufferedInputFile(
                    author_image_cache[top_author_eng],
                    filename=f"{top_author_eng}.jpg"
                ),
                caption=f"Скорее всего автор: {writers_dict[top_author_eng]}"
            )

        # Отправляем текстовый ответ
        await processing_msg.edit_text('\n'.join(response_text))

        # Сообщение о генерации анекдота
        if top_author_rus in JOKE_DF['author'].unique():
            await message.reply(f"⏳ Подождите, я напишу анекдот про автора")

        # Проверка наличия анекдотов
        joke_available = False
        if top_author_rus in JOKE_DF['author'].unique():
            # Получаем строки с анекдотами для автора
            author_jokes = JOKE_DF[JOKE_DF['author'] == top_author_rus]
            if not author_jokes.empty:
                # Получаем список анекдотов для автора
                jokes_list = author_jokes['jokes'].iloc[0]

                if len(jokes_list) > 0:
                    user_jokes = shown_jokes.get(message.from_user.id, {}).get(top_author_rus, [])
                    remaining_jokes = [joke for joke in jokes_list if joke not in user_jokes]

                    if not remaining_jokes:
                        if message.from_user.id in shown_jokes and top_author_rus in shown_jokes[message.from_user.id]:
                            shown_jokes[message.from_user.id][top_author_rus] = []
                        remaining_jokes = jokes_list

                    if len(remaining_jokes) > 0:
                        selected_joke = remaining_jokes[0]

                        # Сохранение в кэш
                        if message.from_user.id not in shown_jokes:
                            shown_jokes[message.from_user.id] = {}
                        if top_author_rus not in shown_jokes[message.from_user.id]:
                            shown_jokes[message.from_user.id][top_author_rus] = []
                        shown_jokes[message.from_user.id][top_author_rus].append(selected_joke)

                        # Формирование анекдота
                        full_joke = f"Анекдот про {top_author_rus}:\n{selected_joke}"

                        # Генерация изображения
                        image_prompt = sanitize_prompt(f"Анекдот про {top_author_rus}: {selected_joke}"[:200])
                        pipeline_id = await fb_api.get_pipeline()
                        uuid = await fb_api.generate_image(image_prompt, pipeline_id)

                        # Проверка генерации изображения
                        if files := await fb_api.check_generation(uuid):
                            file_url = files[0]
                            data_part = file_url.split(',', 1)[1] if file_url.startswith('data:image') else file_url
                            image_data = base64.b64decode(data_part)

                            # Отправка анекдота и изображения
                            await message.reply_photo(
                                photo=BufferedInputFile(image_data, "joke_image.jpg"),
                                caption=full_joke[:800]
                            )
                        else:
                            await message.reply(full_joke)
                        joke_available = True

    except Exception as e:
        logger.error(f"Ошибка: {str(e)}", exc_info=True)
        await message.reply("❌ Произошла ошибка при анализе текста")
    finally:
        await fb_api.close()
        await state.set_state(Form.choosing_function)
        await message.answer(
            "Выберите следующее действие:",
            reply_markup=FUNCTION_KEYBOARD
        )
        active_users.discard(message.from_user.id)


@router.message(Form.waiting_for_author)
async def handle_author_selection(message: types.Message, state: FSMContext):
    selected_author = None
    selected_author_rus = None

    for eng, rus in writers_dict.items():
        if rus == message.text:
            selected_author = eng
            selected_author_rus = rus
            break
    if not selected_author:
        logger.error(f"Автор {message.text} не найден")
        return await message.reply("Неизвестный автор. Пожалуйста, выберите из списка:")
    await state.update_data(selected_author=selected_author)
    await state.set_state(Form.waiting_for_text)
    return await message.reply(
        f"Выбран автор: {selected_author_rus}\n\n"
        "Введите текст на английском языке для его продолжения в стиле выбранного автора"
    )


# Обработчик генерации текста
@router.message(Form.waiting_for_text)
async def handle_text_generation(message: types.Message, state: FSMContext):
    if not message.text or message.text.strip() == "" or message.text in list(writers_dict.values()):
        return await message.reply("⚠️ Пожалуйста, введите текст на английском для его продолжения в стиле выбранного автора")

    if message.from_user.id in active_users:
        return await message.reply("⏳ Пожалуйста, дождитесь предыдущей операции")

    active_users.add(message.from_user.id)
    processing_msg = None

    try:
        data = await state.get_data()
        author = data['selected_author']
        prompt = message.text[:4096]
        translated_text = ''

        # Автоматический перевод текста на английский
        src_lang = detect(prompt)
        if src_lang != 'en':
            translated_text = await translate_text(prompt, src_lang=src_lang)
            if src_lang == 'ru':
                await message.reply(f"🔄 Текст переведён с русского на английский для анализа.")
                await message.reply(translated_text)

        processing_msg = await bot.send_message(
            message.chat.id,
            "🧠 Генерирую текст и изображение..."
        )

        if translated_text != '':
            generated_text = await generate_text(author, translated_text)
        else:
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

        # await message.reply(f"✍️ Текст в стиле {writers_dict[author]}:\n\n{generated_text[:800]}")
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
