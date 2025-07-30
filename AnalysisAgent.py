import json
from typing import List, Dict, Any
from langchain_gigachat.chat_models import GigaChat
import os
from dotenv import load_dotenv

load_dotenv()

class AnalysisAgent:
    """
    Агент анализа, который объединяет результаты от всех компонентов
    и формирует финальный ответ с обоснованием 
    """
    
    def __init__(self):
        self.api_key = os.getenv("GIGACHAT_API_KEY")
        self.model = "GigaChat:latest"

    def _call_gigachat(self, messages: List[Dict[str, str]]) -> str:
        giga = GigaChat(
            credentials=self.api_key,
            verify_ssl_certs=False,
            model=self.model
        )
        return giga.invoke(messages).content
    
    def synthesize_answer(self, 
                         question: str,
                         table_results: List[Dict[str, Any]],
                         text_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Синтезирует финальный ответ на основе всех промежуточных результатов
        
        Args:
            question: Исходный вопрос
            table_results: Результаты поиска в таблицах  
            text_results: Результаты поиска в текстах
            
        Returns:
            Финальный ответ в формате JSON с reasoning
        """
        
        reasoning_steps = []
        
        # Обрабатываем результаты из таблиц
        for result in table_results:
            if result['status'] == 'completed':
                for output in result['output']:
                    reasoning_steps.append({
                        "step": "table_search",
                        "action": "Поиск в таблице",
                        "finding": output.get('summary', 'No summary available'),
                        "source": "table"
                    })
        
        # Обрабатываем результаты из текстов
        for result in text_results:
            if result['status'] == 'completed':
                for output in result['output']:
                    reasoning_steps.append({
                        "step": "text_search",
                        "action": "Поиск в тексте",
                        "finding": output.get('summary', 'No summary available'),
                        "source": "text"
                    })
        
        # Формируем сообщение для GigaChat
        system_prompt = (
            "Ты — агент анализа. Объедини результаты из таблиц и текстов, "
            "чтобы сформировать финальный ответ на вопрос. "
            "Ответь в формате JSON с reasoning."
        )
        
        user_prompt = f"""
                        Вопрос: {question}

                        Результаты поиска:
                        Таблицы: {json.dumps(table_results, ensure_ascii=False, indent=2)}
                        Тексты: {json.dumps(text_results, ensure_ascii=False, indent=2)}

                        Сформируй финальный ответ.
                        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self._call_gigachat(messages)
        
        return response
