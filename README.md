# Multi-Agent System

Многоагентная система для обработки сложных вопросов из датасета HybridQA.

## Файлы

- `app.py` - FastAPI сервер с REST API
- `main.py` - Логика многоагентной системы
- `PlannerLLM.py` - Агент планирования рассуждения
- `TableAgent.py` - Агент поиска в табличных данных
- `rag.py` - RAG агент для поиска контекста по статьям wiki
- `AnalysisAgent.py` - Агент синтеза финального ответа
- `preprocessed_data/` - JSON данные для работы системы

## Технологии

- **GigaChat** - LLM для генерации ответов
- **LangChain** - Фреймворк для работы с LLM
- **FastAPI** - REST API сервер
- **FAISS** - Векторная база данных для RAG

## Запуск
установить библиотеки requirements.txt
Создать .env файл с GIGACHAT_API_KEY=your_key
в preprocessed_data/ скачать данные
```bash
wget https://hybridqa.s3-us-west-2.amazonaws.com/preprocessed_data.zip
unzip preprocessed_data.zip
```
чтобы создать индекс раскоментировать внизу rag.py и запустить
```bash
python rag.py
```
Запустить приложение
```bash
python app.py
```
или Docker
```bash
docker-compose up -d
```

API доступен по адресу: http://localhost:8000/docs
