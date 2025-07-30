from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import uvicorn
import json
from datetime import datetime

# Импортируем нашу многоагентную систему
from main import MultiAgentSystem

# Инициализируем FastAPI приложение
app = FastAPI(
    title="Multi-Agent System API",
    description="API для многоагентной системы обработки вопросов",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Глобальная переменная для системы
system = None

# Модели данных
class QuestionRequest(BaseModel):
    question: str = Field('What lemur was classified in 2015 and has a conservation status that has not been evaluated and has an average size over 200 ?',
                           description="Вопрос, который нужно обработать")
    use_rag: Optional[bool] = True

class QuestionResponse(BaseModel):
    status: str
    question: str
    answer: str
    reasoning: str
    confidence: float
    function_calls: list
    processing_time_seconds: float
    session_id: str
    agents_involved: list
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    details: str
    timestamp: str

@app.on_event("startup")
async def startup_event():
    """Инициализация системы при запуске сервера"""
    global system
    try:
        system = MultiAgentSystem("preprocessed_data")
    except Exception as e:
        print(f"Ошибка инициализации системы: {e}")
        raise

@app.get("/")
async def root():
    """Корневой эндпоинт с информацией об API"""
    return {
        "message": "Multi-Agent System API",
        "version": "1.0.0",
        "status": "ready" if system else "not_initialized",
        "endpoints": {
            "/ask": "POST - Задать вопрос системе",
            "/health": "GET - Проверка состояния системы",
            "/stats": "GET - Статистика системы",
            "/docs": "GET - Swagger документация",
            "/redoc": "GET - ReDoc документация"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка состояния системы"""
    if not system:
        raise HTTPException(status_code=503, detail="Система не инициализирована")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_initialized": True,
        "rag_available": hasattr(system.rag_agent, 'vectorstore') and system.rag_agent.vectorstore is not None,
        "data_loaded": len(system.data_cache.get("questions", [])) > 0
    }

@app.get("/stats")
async def get_system_stats():
    """Получение статистики системы"""
    if not system:
        raise HTTPException(status_code=503, detail="Система не инициализирована")
    
    return {
        "questions_loaded": len(system.data_cache.get("questions", [])),
        "tables_loaded": len(system.data_cache.get("tables", {})),
        "nodes_loaded": sum(len(nodes) for nodes in system.data_cache.get("nodes_by_question", {}).values()),
        "rag_status": "available" if hasattr(system.rag_agent, 'vectorstore') and system.rag_agent.vectorstore else "unavailable",
        "session_id": system.session_id,
        "data_folder": system.data_folder
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Основной эндпоинт для обработки вопросов
    
    Принимает вопрос и возвращает ответ от многоагентной системы
    """
    if not system:
        raise HTTPException(status_code=503, detail="Система не инициализирована")
    
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")
    
    try:
        print(f"\n📝 Получен вопрос: {request.question}")
        
        # Обрабатываем вопрос через многоагентную систему
        result = system.process_question(
            question=request.question.strip(),
            use_rag=request.use_rag
        )
        
        # Добавляем timestamp к ответу
        result["timestamp"] = datetime.now().isoformat()
        
        print(f"Ответ готов за {result.get('processing_time_seconds', 0)} сек")
        
        return QuestionResponse(**result)
        
    except Exception as e:
        print(f"Ошибка обработки вопроса: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Ошибка обработки вопроса: {str(e)}"
        )

# Обработчик ошибок
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Общий обработчик исключений"""
    return {
        "error": "Internal Server Error",
        "details": str(exc),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 