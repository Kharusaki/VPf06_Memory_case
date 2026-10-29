"""
Telegram-бот продавец учебных курсов по английскому языку с краткосрочной и долгосрочной памятью (RAG).

Технологии:
- aiogram 3.x (асинхронные хендлеры)
- LangChain (LCEL) для построения цепочки ответа
- OpenAI ChatCompletion API через proxyapi (gpt-4.1-mini)
- OpenAI Embeddings (text-embedding-3-small)
- ChromaDB с персистентным хранением в ./memory
- Разбивка документов на чанки ~500 символов

Функции по ТЗ:
- load_document()   - загрузка и извлечение текста из PDF/TXT/DOCX
- embed_chunks()    - разбиение текста на чанки и сохранение эмбеддингов в Chroma
- retrieve_context()- поиск релевантных фрагментов в векторной БД
- answer_question() - формирование ответа через LangChain цепочку с контекстом + историей
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv

# aiogram
from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.types import Message, FSInputFile
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Парсеры
from pypdf import PdfReader
from docx import Document as DocxDocument

# Загрузка переменных окружения
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/v1")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-5-mini")
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY")

# Константы
UPLOAD_DIR = Path("./uploads")
MEMORY_DIR = Path("./memory")
HISTORY_LIMIT_MESSAGES = 10  # хранить последние 10 сообщений на user_id (как в ТЗ)

# Создаем необходимые директории
UPLOAD_DIR.mkdir(exist_ok=True)
MEMORY_DIR.mkdir(exist_ok=True)

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Краткосрочная память в dict по user_id
# Формат: {user_id: [{"role": "user"|"assistant", "content": "..."}, ...]}
short_term_memory: Dict[int, List[Dict[str, str]]] = {}

# Инициализация бота и диспетчера
session = None
if TELEGRAM_PROXY:
    session = AiohttpSession(proxy=TELEGRAM_PROXY)
    logger.info(f"Используется прокси для Telegram: {TELEGRAM_PROXY}")

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    session=session
)
dp = Dispatcher()
router = Router()
dp.include_router(router)

# Инициализация компонентов LangChain
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL,
)

llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL,
    # temperature не поддерживается для gpt-5-mini через proxyapi
)

# Векторное хранилище Chroma
vectorstore = Chroma(
    persist_directory=str(MEMORY_DIR),
    embedding_function=embeddings,
    collection_name="english_courses_docs"
)

# === ФУНКЦИИ ПО ТЗ ===

def is_history_question(text: str) -> bool:
    """Определяет вопросы про историю диалога, на которые нужно отвечать из краткосрочной памяти"""
    t = text.lower().replace("ё", "е")
    
    # Если вопрос про содержимое документа - НИКОГДА не идём в историю
    doc_words = [
        "тариф", "тарифы", "курс", "курсы", "цена", "стоимость", "скидк",
        "рассроч", "промокод", "документ", "бизнес", "ielts", "speakup",
        "beginner", "zero", "с нуля", "разговорн", "английск", "faq", "уровень"
    ]
    if any(w in t for w in doc_words):
        return False
    
    # Вопросы про сам диалог
    if "роль" in t and "user" in t:
        return True
    if "только те сообщения" in t:
        return True
    if "что мы обсуждали" in t or "о чем мы говорили" in t or "о чём мы говорили" in t:
        return True
    if "истори" in t and "диалог" in t:
        return True
    if "напомни" in t and ("вопрос" in t or "диалог" in t or "общение" in t):
        return True
    if ("перечисли" in t) and ("вопрос" in t):
        return True
    if ("какие вопросы" in t) and ("задавал" in t or "задавала" in t or "спрашивал" in t or "спрашивала" in t):
        return True
    if "задавал ли я" in t or "задавала ли я" in t:
        return True
    if "помнишь ли" in t and ("вопрос" in t or "spaasial" not in t and "задавал" in t):
        return True
    
    return False


def load_document(file_path: str) -> str:
    """
    Загрузка и извлечение текста из файла PDF/TXT/DOCX.
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        str: Извлеченный текст из документа
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    text = ""
    
    try:
        if ext == ".txt":
            # Загрузка текстового файла
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            logger.info(f"Загружен TXT файл: {path.name}, символов: {len(text)}")
        
        elif ext == ".pdf":
            # Загрузка PDF файла
            reader = PdfReader(path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            logger.info(f"Загружен PDF файл: {path.name}, страниц: {len(reader.pages)}, символов: {len(text)}")
        
        elif ext == ".docx":
            # Загрузка DOCX файла
            doc = DocxDocument(path)
            for para in doc.paragraphs:
                text += para.text + "\n"
            logger.info(f"Загружен DOCX файл: {path.name}, абзацев: {len(doc.paragraphs)}, символов: {len(text)}")
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {ext}. Поддерживаются: PDF, TXT, DOCX")
        
        return text.strip()
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке документа {file_path}: {e}")
        raise

def embed_chunks(text: str, source: str = "uploaded_document") -> int:
    """
    Разбивка текста на части (~650 символов) и сохранение в ChromaDB.
    
    Args:
        text: Текст для разбивки
        source: Источник документа
        
    Returns:
        int: Количество добавленных чанков
    """
    try:
        # Более умная разбивка с приоритетом по абзацам/блокам курсов
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=40,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        chunks = text_splitter.split_text(text)
        logger.info(f"Текст разбит на {len(chunks)} чанков (chunk_size≈800)")
        
        # Создание документов с метаданными
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": source, "chunk_id": i}
                )
            )
        
        # Сохранение в ChromaDB (персистентное хранение)
        if documents:
            vectorstore.add_documents(documents)
            logger.info(f"Добавлено {len(documents)} чанков в ChromaDB (папка {MEMORY_DIR})")
            return len(documents)
        
        return 0
    
    except Exception as e:
        logger.error(f"Ошибка при векторизации чанков: {e}")
        raise

def retrieve_context(query: str, k: int = 8) -> str:
    """
    Поиск релевантных фрагментов в векторной базе.
    
    Args:
        query: Поисковый запрос пользователя
        k: Количество релевантных чанков для поиска
        
    Returns:
        str: Собранный контекст из найденных фрагментов
    """
    try:
        # Поиск похожих документов
        docs = vectorstore.similarity_search(query, k=k)
        
        if not docs:
            logger.info("Релевантных фрагментов в векторной БД не найдено")
            return ""
        
        # Сбор контекста
        context_parts = []
        for i, doc in enumerate(docs):
            context_parts.append(f"[Фрагмент {i+1}]\n{doc.page_content}\n")
        
        context = "\n".join(context_parts)
        logger.info(f"Найдено {len(docs)} релевантных фрагментов, длина контекста: {len(context)} символов")
        return context
    
    except Exception as e:
        logger.error(f"Ошибка при поиске в векторной БД: {e}")
        return ""

def answer_question(user_id: int, question: str) -> str:
    """
    Формирование ответа с использованием короткой истории + контекста из RAG.
    Использует LangChain chains.
    
    Args:
        user_id: ID пользователя
        question: Вопрос пользователя
        
    Returns:
        str: Ответ модели
    """
    try:
        # Получаем короткую историю диалога (последние 10 сообщений)
        history = short_term_memory.get(user_id, [])
        # Обрезаем до HISTORY_LIMIT_MESSAGES (последние 10)
        if len(history) > HISTORY_LIMIT_MESSAGES:
            history = history[-HISTORY_LIMIT_MESSAGES:]
            short_term_memory[user_id] = history
        
        # Формируем историю в виде текста
        history_text = ""
        for msg in history:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            history_text += f"{role}: {msg['content']}\n\n"
        
        # Поиск релевантного контекста в векторной БД (RAG)
        context = retrieve_context(question, k=8)
        
        # Системный промпт для бота-продавца курсов по английскому языку
        system_prompt = """Ты - бот-продавец учебных курсов по английскому языку EnglishPro.

ПРАВИЛА ОТВЕТА:
- Отвечай ТОЛЬКО на основе предоставленного КОНТЕКСТА из документа с информацией о курсах.
- Если пользователь просит ПЕРЕЧИСЛИТЬ ВСЕ КУРСЫ/ТАРИФЫ - проанализируй ВСЕ фрагменты контекста целиком и перечисли КАЖДЫЙ курс, который в них упоминается. Не выбирай только "самые релевантные".
- Если в контексте упомянуты разные курсы - обязательно выведи все 4 курса, если они есть в документе.
- Если в разных фрагментах встречаются описание и ценник одного и того же курса (совпадают цена, формат, продолжительность, описание) - ОБЯЗАТЕЛЬНО объедини их в одно описание.
- Если пользователь просит ПЕРЕЧИСЛИТЬ КУРСЫ/ТАРИФЫ - дай ТОЛЬКО информацию по курсам: название, уровень, продолжительность, формат, стоимость. НЕ добавляй акции, промокоды, рассрочку, если их прямо не просили.
- Если пользователь прямо спрашивает про акцию/скидку/рассрочку/промокод - можешь ответить по этим пунктам.
- НИКОГДА не упоминай про чанки, фрагменты, разбивку или то, что какой-то блок "оторвался".
- НЕ выдумывай названия курсов, цены, акции, условия и другую информацию.
- Отвечай кратко, по делу, на языке пользователя.
- Если информации нет в контексте - честно скажи: "В загруженном документе нет информации по этому вопросу."
- Если вопрос не относится к курсам английского языка и информации из документа - скажи, что можешь отвечать только по информации из загруженного документа."""

        # Промпт для LangChain цепочки
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """КОНТЕКСТ ИЗ ДОКУМЕНТА:
{context}

ИСТОРИЯ ДИАЛОГА (короткая память):
{history}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}

Ответь на вопрос, строго опираясь только на контекст из документа.""")
        ])
        
        # Создание цепочки через LangChain LCEL
        chain = prompt_template | llm | StrOutputParser()
        
        # Генерация ответа
        answer = chain.invoke({
            "context": context if context else "КОНТЕКСТ НЕ НАЙДЕН. В векторной базе нет релевантной информации.",
            "history": history_text if history_text else "История диалога пуста.",
            "question": question
        })
        
        logger.info(f"Сгенерирован ответ для user_id={user_id}, длина: {len(answer)}")
        return answer.strip()
    
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {e}")
        return "Произошла ошибка при генерации ответа. Попробуйте позже."

# === ХЕНДЛЕРЫ AIROGRAM ===

@router.message(F.text == "/start")
async def cmd_start(message: Message):
    """Обработчик команды /start"""
    user_id = message.from_user.id
    # Инициализируем историю для пользователя если её нет
    if user_id not in short_term_memory:
        short_term_memory[user_id] = []
    
    welcome_text = """<b>Привет! Я бот-продавец учебных курсов по английскому языку EnglishPro</b>

<b>Что я могу:</b>
• Консультировать по курсам английского языка на основе загруженного документа
• Помнить контекст нашего диалога (последние 10 сообщений)
• Искать нужную информацию в базе знаний

<b>Как начать работу:</b>
1. Загрузите файл с информацией о курсах в формате <b>PDF</b>, <b>TXT</b> или <b>DOCX</b>
2. После загрузки и индексации задавайте вопросы по документу

<b>Команды:</b>
/start - начать работу
/clear - очистить историю диалога
/status - проверить статус бота"""
    
    await message.answer(welcome_text)

@router.message(F.text == "/clear")
async def cmd_clear(message: Message):
    """Очистка короткой истории диалога"""
    user_id = message.from_user.id
    short_term_memory[user_id] = []
    await message.answer("История диалога очищена.")

@router.message(F.text == "/status")
async def cmd_status(message: Message):
    """Проверка статуса бота"""
    user_id = message.from_user.id
    history_count = len(short_term_memory.get(user_id, []))
    
    # Проверяем количество документов в векторной БД
    try:
        collection_count = vectorstore._collection.count()
    except:
        collection_count = 0
    
    status_text = f"""<b>Статус бота:</b>

• Короткая память (история): {history_count} / {HISTORY_LIMIT_MESSAGES} сообщений
• Документов в векторной БД (Chroma): {collection_count} чанков
• Папка с памятью: {MEMORY_DIR}
• Папка для загрузок: {UPLOAD_DIR}
• Модель: {DEFAULT_MODEL}
• Режим: RAG (только по загруженному документу)"""
    
    await message.answer(status_text)

@router.message(F.document)
async def handle_document(message: Message):
    """Обработчик загрузки документов PDF/TXT/DOCX"""
    user_id = message.from_user.id
    
    try:
        document = message.document
        file_name = document.file_name
        file_ext = Path(file_name).suffix.lower()
        
        # Проверка формата файла
        if file_ext not in [".pdf", ".txt", ".docx"]:
            await message.answer("Поддерживаются только файлы в форматах <b>PDF</b>, <b>TXT</b> или <b>DOCX</b>.")
            return
        
        await message.answer(f"Загружаю файл <b>{file_name}</b>...")
        
        # Скачивание файла
        file_info = await bot.get_file(document.file_id)
        file_path = UPLOAD_DIR / file_name
        await bot.download_file(file_info.file_path, destination=file_path)
        
        await message.answer("Обрабатываю документ...")
        
        # Загрузка текста из документа
        text = load_document(str(file_path))
        
        if not text:
            await message.answer("Не удалось извлечь текст из документа. Проверьте файл.")
            return
        
        await message.answer(f"Текст извлечен. Символов: {len(text)}. Разбиваю на чанки...")
        
        # Векторизация и сохранение в ChromaDB
        num_chunks = embed_chunks(text, source=file_name)
        
        await message.answer(
            f"✅ Документ успешно обработан!\n\n"
            f"Файл: <b>{file_name}</b>\n"
            f"Создано чанков: <b>{num_chunks}</b>\n"
            f"Хранение: <b>{MEMORY_DIR}/</b>\n\n"
            f"Теперь вы можете задавать вопросы по этому документу."
        )
        
        logger.info(f"Пользователь {user_id} загрузил и проиндексировал файл {file_name}")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке документа: {e}")
        await message.answer(f"Произошла ошибка при обработке файла:\n{str(e)}")

@router.message(F.text & ~F.text.startswith("/"))
async def handle_question(message: Message):
    """Обработчик текстовых вопросов пользователя"""
    user_id = message.from_user.id
    question = message.text.strip()
    
    try:
        # Инициализируем историю если её нет
        if user_id not in short_term_memory:
            short_term_memory[user_id] = []

        # Если вопрос про историю диалога - отвечаем ТОЛЬКО из краткосрочной памяти
        if is_history_question(question):
            history = short_term_memory.get(user_id, [])
            # Берём только пользовательские сообщения
            user_questions = []
            for msg in history:
                if msg["role"] == "user":
                    user_questions.append(msg["content"])
            
            if not user_questions:
                answer = "У нас пока нет истории диалога. Ты ещё ничего не спрашивал."
            else:
                # Отвечаем строго по запросу: перечисляем только вопросы пользователя
                answer = "\n".join(user_questions)
            
            # Сохраняем пару вопрос-ответ в краткосрочную память
            short_term_memory[user_id].append({"role": "user", "content": question})
            short_term_memory[user_id].append({"role": "assistant", "content": answer})
            
            if len(short_term_memory[user_id]) > HISTORY_LIMIT_MESSAGES:
                short_term_memory[user_id] = short_term_memory[user_id][-HISTORY_LIMIT_MESSAGES:]
            
            await message.answer(answer)
            logger.info(f"Обработан исторический вопрос от user_id={user_id}")
            return

        # Иначе - обычный вопрос по документу (RAG)
        await message.answer("Ищу ответ в документе...")
        
        # Получаем ответ с использованием RAG + истории диалога
        answer = answer_question(user_id, question)
        
        # Сохраняем вопрос и ответ в короткую память (последние 10 сообщений)
        short_term_memory[user_id].append({"role": "user", "content": question})
        short_term_memory[user_id].append({"role": "assistant", "content": answer})
        
        # Обрезаем до 10 сообщений
        if len(short_term_memory[user_id]) > HISTORY_LIMIT_MESSAGES:
            short_term_memory[user_id] = short_term_memory[user_id][-HISTORY_LIMIT_MESSAGES:]
        
        await message.answer(answer)
        logger.info(f"Обработан вопрос от user_id={user_id}")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке вопроса: {e}")
        await message.answer("Произошла ошибка. Попробуйте снова.")

async def main():
    """Главная функция запуска бота"""
    logger.info("Запуск бота...")
    logger.info(f"Модель: {DEFAULT_MODEL}")
    logger.info(f"ChromaDB: {MEMORY_DIR}")
    logger.info(f"История диалога: {HISTORY_LIMIT_MESSAGES} сообщений")
    if TELEGRAM_PROXY:
        logger.info(f"Telegram прокси включен")
    
    # Проверка наличия токенов
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN не найден в .env")
        return
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY не найден в .env")
        return
    
    try:
        # Запуск бота
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")
        raise
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
