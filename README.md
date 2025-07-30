# Multi-Agent System

Многоагентная система для обработки сложных вопросов с использованием GigaChat + LangChain.

## Файлы

- `app.py` - FastAPI сервер с REST API
- `main.py` - Логика многоагентной системы 
- `PlannerLLM.py` - Агент планирования рассуждения
- `TableAgent.py` - Агент поиска в табличных данных
- `rag.py` - RAG агент для поиска в текстах
- `AnalysisAgent.py` - Агент синтеза финального ответа
- `preprocessed_data/` - JSON данные для работы системы

## Технологии

- **GigaChat** - LLM для генерации ответов
- **LangChain** - Фреймворк для работы с LLM
- **FastAPI** - REST API сервер
- **FAISS** - Векторная база данных для RAG

## Запуск

```bash
# Создать .env файл с GIGACHAT_API_KEY=your_key
docker-compose up -d
```

API доступен по адресу: http://localhost:8000/docs 