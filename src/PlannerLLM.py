import json
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat

load_dotenv()


class PlannerLLM:
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

    def create_reasoning_plan(self, question: str, table_id: Optional[str] = None) -> Dict[str, Any]:
        system_prompt = (
            "Ты — агент-планировщик для сложных вопросов. "
            "Разбей задачу на шаги с уникальными переменными (#E1, #E2 и т.д.). "
            "Всегда используй table_search первым шагом"
            "Варианты action_type: table_search, rag_search, analysis"
            "Каждый шаг должен быть в формате:\n"
            "{\n"
            '  "id": "#E1",\n'
            '  "action_type": "table_search" ,\n'
            '  "description": "что сделать",\n'
            '  "input": "если используется результат предыдущих шагов",\n'
            '  "expected_output": "что получить"\n'
            "}\n"
            "{\n"
            '  "id": "#E4",\n'
            '  "action_type": "rag_search",\n'
            '  "description": "поиск ближайшей статьи в RAG",\n'
            '  "input": "если используется результат предыдущих шагов",\n'
            '  "expected_output": "что получить"\n'
            "}\n"
            "{\n"
            '  "id": "#E9",\n'
            '  "action_type": "analysis",\n'
            '  "description": "анализ результатов и формирование ответа",\n'
            '  "input": "результаты предыдущих шагов",\n'
            '  "expected_output": "финальный ответ"\n'
            "}\n"
            "Верни результат строго как JSON:\n"
            "{\n"
            '  "reasoning_steps": [...],\n'
            '  "final_goal": "..." \n'
            "}"
        )

        user_prompt = f"""
                        Вопрос: {question}
                        Таблица: {table_id or "неизвестна"}

                        Сформируй план рассуждения с переменными и шагами.
                        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self._call_gigachat(messages)

        try:
            plan = json.loads(response)
            return {
                "question": question,
                "table_id": table_id,
                "plan": plan,
                "status": "success"
            }
        except json.JSONDecodeError:
            return {
                "question": question,
                "table_id": table_id,
                "plan": {"error": "JSON parse error", "raw_response": response},
                "status": "error"
            }


# planner = PlannerLLM()
# plan_result = planner.create_reasoning_plan("What is the population of the hometown of the 2012 Gatorade Player of the Year ?")
# print(plan_result)