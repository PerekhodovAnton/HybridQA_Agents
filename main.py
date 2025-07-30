import os
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

# Импортируем всех агентов
from PlannerLLM import PlannerLLM
from TableAgent import TableToolAgent, load_json_data
from rag import RetrievalAgent
from AnalysisAgent import AnalysisAgent


class MultiAgentSystem:
    """
    Главный класс многоагентной системы, координирующий работу всех агентов
    """
    
    def __init__(self, data_folder: str = "preprocessed_data"):
        self.data_folder = data_folder
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Инициализируем агентов
        print("Инициализация многоагентной системы...")
        
        try:
            self.planner = PlannerLLM()
            self.table_agent = TableToolAgent()
            self.rag_agent = RetrievalAgent(data_folder)
            self.analysis_agent = AnalysisAgent()
            
            # Загружаем FAISS индекс для RAG агента
            try:
                from langchain_community.vectorstores import FAISS
                self.rag_agent.vectorstore = FAISS.load_local("my_index", self.rag_agent.embedder, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"е удалось загрузить FAISS индекс: {e}")
                print("RAG функциональность будет недоступна")
                self.rag_agent.vectorstore = None
            
            self.data_cache = self._load_all_data()
            
            print("Система инициализирована успешно!")
            
        except Exception as e:
            print(f"Ошибка инициализации: {e}")
            raise
    
    def _load_all_data(self) -> Dict[str, Any]:
        """Загружает все доступные данные для работы агентов"""
        data_cache = {
            "questions": [],
            "tables": {},
            "nodes_by_question": {}
        }
        
        try:
            import glob
            json_files = glob.glob(os.path.join(self.data_folder, "*.json"))
            
            for json_file in json_files:
                file_name = os.path.basename(json_file)
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for record in data:
                    question_id = record.get('question_id', 'unknown')
                    
                    data_cache["questions"].append({
                        "question_id": question_id,
                        "question": record.get('question', ''),
                        "table_id": record.get('table_id', ''),
                        "source_file": file_name
                    })
                    
                    data_cache["nodes_by_question"][question_id] = record.get('nodes', [])
                    
                    if 'table' in record:
                        data_cache["tables"][question_id] = record['table']
            
            print(f"Загружено {len(data_cache['questions'])} вопросов")
            
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
        
        return data_cache
    
    def process_question(self, question: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        Главный метод обработки вопроса через всю систему агентов
        """
        
        start_time = time.time()
        session_log = {
            "session_id": self.session_id,
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "final_result": None,
            "processing_time": 0,
            "errors": []
        }
        
        try:
            # Создание плана рассуждения
            plan_result = self._create_reasoning_plan(question)
            session_log["steps"].append({
                "step": 1,
                "agent": "PlannerLLM", 
                "action": "create_reasoning_plan",
                "result": plan_result,
                "status": plan_result.get("status", "unknown")
            })
            
            if plan_result.get("status") != "success":
                raise Exception(f"Не удалось создать план: {plan_result}")
            
            # Выполнение шагов плана
            step_results = self._execute_plan_steps(question, plan_result)
            session_log["steps"].extend(step_results)
            
            # Синтез финального ответа (только если план не содержит analysis шаг)
            reasoning_steps = plan_result.get("plan", {}).get("reasoning_steps", [])
            has_analysis_step = any(step.get("action_type") == "analysis" for step in reasoning_steps)
            
            if has_analysis_step:
                # Если план уже содержит analysis шаг, извлекаем результат из него
                final_answer = "Результат анализа"
                for step_result in step_results:
                    if step_result.get("action") == "analysis":
                        final_answer = step_result.get("result", {}).get("output", "Результат анализа")
                        break
            else:
                # Если плана нет analysis шага, выполняем синтез
                final_answer = self._synthesize_final_answer(question, step_results)
                session_log["steps"].append({
                    "step": len(session_log["steps"]) + 1,
                    "agent": "AnalysisAgent",
                    "action": "synthesize_answer",
                    "result": final_answer,
                    "status": "completed"
                })
            
            # Формирование JSON ответа
            json_result = self._format_json_response(question, session_log, final_answer)
            
            session_log["final_result"] = json_result
            session_log["processing_time"] = round(time.time() - start_time, 2)
            
            return json_result
            
        except Exception as e:
            error_msg = f"Ошибка обработки: {str(e)}"
            session_log["errors"].append({
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            })
            
            session_log["processing_time"] = round(time.time() - start_time, 2)
            
            return {
                "status": "error",
                "question": question,
                "answer": "Не удалось найти ответ из-за технической ошибки",
                "reasoning": f"Система столкнулась с ошибкой: {error_msg}",
                "confidence": 0.0,
                "function_calls": [],
                "processing_time_seconds": session_log["processing_time"],
                "session_id": session_log["session_id"],
                "agents_involved": [],
                "session_log": session_log
            }
    
    def _create_reasoning_plan(self, question: str) -> Dict[str, Any]:
        """Создает план рассуждения через PlannerLLM"""
        try:
            # Поиск подходящего table_id
            table_id = None
            question_words = set(question.lower().split())
            
            for q_data in self.data_cache["questions"]:
                stored_words = set(q_data["question"].lower().split())
                overlap = len(question_words & stored_words)
                if overlap > 2:  # Найдены пересечения
                    table_id = q_data["table_id"]
                    break
            
            plan_result = self.planner.create_reasoning_plan(question, table_id)
            reasoning_steps = plan_result.get('plan', {}).get('reasoning_steps', [])
            
            for step in reasoning_steps:
                print(f"    {step.get('id', '?')}: {step.get('action_type', '?')} - {step.get('description', '?')}")
            
            return plan_result
            
        except Exception as e:
            print(f"Ошибка создания плана: {e}")
            return {"status": "error", "error": str(e)}
    
    def _execute_plan_steps(self, question: str, plan_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Выполняет все шаги плана согласно их action_type"""
        all_step_results = []
        step_outputs = {}
        
        try:
            reasoning_steps = plan_result.get("plan", {}).get("reasoning_steps", [])
            
            for step in reasoning_steps:
                step_id = step.get("id")
                action_type = step.get("action_type")
                description = step.get("description", "")
                
                try:
                    if action_type == "rag_search":
                        result = self._execute_rag_search_step(question, step, step_outputs)
                    elif action_type == "table_search":
                        result = self._execute_table_search_step(question, step, step_outputs)
                    elif action_type == "analysis":
                        result = self._execute_analysis_step(question, step, step_outputs)
                    else:
                        print(f"Неизвестный action_type: {action_type}")
                        result = {
                            "step_id": step_id,
                            "status": "error",
                            "error": f"Unknown action_type: {action_type}",
                            "output": None
                        }
                    
                    step_outputs[step_id] = result.get("output")
                    
                    step_result = {
                        "step": len(all_step_results) + 2,
                        "agent": self._get_agent_name_by_action_type(action_type),
                        "action": action_type,
                        "step_id": step_id,
                        "description": description,
                        "result": result,
                        "status": result.get("status", "unknown")
                    }
                    
                    all_step_results.append(step_result)
                    
                    print(f"{step_id} завершен: {result.get('status', 'unknown')}")
                    
                except Exception as step_error:
                    print(f"Ошибка в шаге {step_id}: {step_error}")
                    
                    error_result = {
                        "step": len(all_step_results) + 2,
                        "agent": self._get_agent_name_by_action_type(action_type),
                        "action": action_type,
                        "step_id": step_id,
                        "description": description,
                        "result": {
                            "step_id": step_id,
                            "status": "error",
                            "error": str(step_error),
                            "output": None
                        },
                        "status": "error"
                    }
                    
                    all_step_results.append(error_result)
                    step_outputs[step_id] = None
            
            return all_step_results
            
        except Exception as e:
            print(f"Критическая ошибка выполнения плана: {e}")
            return []
    
    
    def _execute_rag_search_step(self, question: str, step: Dict[str, Any], step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет шаг поиска через RAG"""
        step_id = step.get("id")
        step_input = step.get("input", "")
        
        try:
            # Формируем поисковый запрос
            search_query = self._build_search_query(question, step, step_outputs)
            
            # Проверяем доступность RAG
            if not hasattr(self.rag_agent, 'vectorstore') or not self.rag_agent.vectorstore:
                return {
                    "step_id": step_id,
                    "status": "completed",
                    "output": [],
                    "summary": "RAG индекс недоступен"
                }
            
            # Выполняем поиск
            search_results = self.rag_agent.retrieve(search_query, top_k=5)
            
            return {
                "step_id": step_id,
                "status": "completed",
                "output": search_results,
                "summary": f"RAG поиск: найдено {len(search_results)} результатов"
            }
            
        except Exception as e:
            return {
                "step_id": step_id,
                "status": "error",
                "error": str(e),
                "output": None
            }
    
    def _execute_table_search_step(self, question: str, step: Dict[str, Any], step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет шаг поиска в таблицах"""
        step_id = step.get("id")
        
        try:
            # Находим релевантные nodes для данного вопроса
            relevant_nodes = self._find_relevant_nodes_for_question(question)
            
            if relevant_nodes:
                result = self.table_agent.execute(
                    step=step,
                    question=question,
                    table_data={"table_id": "universal"},
                    nodes=relevant_nodes
                )
                return result
            else:
                return {
                    "step_id": step_id,
                    "status": "completed",
                    "output": [],
                    "summary": "Релевантные данные в таблицах не найдены"
                }
                
        except Exception as e:
            return {
                "step_id": step_id,
                "status": "error",
                "error": str(e),
                "output": None
            }
    
    def _execute_analysis_step(self, question: str, step: Dict[str, Any], step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет шаг анализа и синтеза"""
        step_id = step.get("id")
        
        try:
            # Собираем все результаты предыдущих шагов
            all_results = []
            for step_id_key, output in step_outputs.items():
                if output:
                    all_results.append({
                        "step_id": step_id_key,
                        "status": "completed",
                        "output": output
                    })
            
            # Вызываем AnalysisAgent
            analysis_result = self.analysis_agent.synthesize_answer(
                question=question,
                table_results=all_results,
                text_results=[]
            )
            
            return {
                "step_id": step_id,
                "status": "completed",
                "output": analysis_result,
                "summary": "Анализ и синтез завершен"
            }
            
        except Exception as e:
            return {
                "step_id": step_id,
                "status": "error", 
                "error": str(e),
                "output": None
            }
    
    def _build_search_query(self, question: str, step: Dict[str, Any], step_outputs: Dict[str, Any]) -> str:
        """Универсально строит поисковый запрос на основе входных данных шага"""
        step_input = step.get("input", "")
        base_query = question
        
        # Обрабатываем ссылки на предыдущие шаги
        import re
        step_refs = re.findall(r'\{выходные данные (#E\d+)\}', step_input)
        
        query_parts = [base_query]
        
        for step_ref in step_refs:
            if step_ref in step_outputs and step_outputs[step_ref]:
                output = step_outputs[step_ref]
                if isinstance(output, list) and len(output) > 0:
                    first_item = output[0]
                    if isinstance(first_item, dict):
                        # Извлекаем текстовые данные из результата
                        text_fields = ['entity', 'text', 'content', 'summary', 'answer']
                        for field in text_fields:
                            if field in first_item:
                                query_parts.append(str(first_item[field]))
                                break
        
        return " ".join(query_parts)
    
    
    def _find_relevant_nodes_for_question(self, question: str) -> List[List[Any]]:
        """Универсально находит релевантные nodes для любого вопроса"""
        question_words = set(word.lower() for word in question.split() if len(word) > 3)
        best_match = None
        best_score = 0
        
        for question_data in self.data_cache["questions"]:
            stored_words = set(word.lower() for word in question_data["question"].split() if len(word) > 3)
            overlap = len(question_words & stored_words)
            
            if overlap > best_score:
                best_score = overlap
                best_match = question_data
        
        if best_match:
            nodes = self.data_cache["nodes_by_question"].get(best_match["question_id"], [])
            return nodes[:10]  # Возвращаем топ-10 наиболее релевантных
        
        return []
    
    def _get_agent_name_by_action_type(self, action_type: str) -> str:
        """Возвращает имя агента по типу действия"""
        agent_mapping = {
            "rag_search": "RetrievalAgent",
            "table_search": "TableAgent",
            "analysis": "AnalysisAgent"
        }
        return agent_mapping.get(action_type, "UnknownAgent")
    
    def _synthesize_final_answer(self, question: str, step_results: List[Dict[str, Any]]) -> str:
        """Универсально синтезирует финальный ответ с дополнительной проверкой качества"""
        try:            
            # Собираем все успешные результаты
            successful_results = []
            for step_result in step_results:
                if step_result.get("status") == "completed":
                    successful_results.append({
                        "step_id": step_result.get("step_id", "unknown"),
                        "status": "completed",
                        "output": step_result.get("result", {}).get("output", []),
                        "summary": step_result.get("result", {}).get("summary", "")
                    })
            
            if not successful_results:
                return "Не удалось получить результаты ни из одного шага"
            
            # Первичный синтез ответа через AnalysisAgent
            try:
                initial_answer = self.analysis_agent.synthesize_answer(
                    question=question,
                    table_results=successful_results,
                    text_results=[]
                )
                initial_answer_str = str(initial_answer)
                
                # Проверяем качество ответа через GigaChat
                print("Проверяем качество первичного ответа...")
                answer_quality = self._check_answer_quality(question, initial_answer_str)
                
                if answer_quality["needs_improvement"]:
                    print(f"Применяем дополнительный RAG поиск для улучшения ответа {answer_quality['reason']}")
                    
                    # Выполняем дополнительный RAG поиск
                    enhanced_answer = self._enhance_answer_with_rag(question, initial_answer_str, successful_results)
                    return enhanced_answer
                else:
                    return initial_answer_str
                
            except Exception as e:
                # Fallback: простое извлечение из последнего результата
                if successful_results:
                    last_result = successful_results[-1]
                    output = last_result.get("output", [])
                    if output and len(output) > 0:
                        if isinstance(output[0], dict) and "answer" in output[0]:
                            return str(output[0]["answer"])
                        elif isinstance(output[0], str):
                            return output[0]
                
                return "Ответ найден, но не удалось его извлечь"
            
        except Exception as e:
            return f"Ошибка синтеза: {str(e)}"
    
    def _check_answer_quality(self, question: str, answer: str) -> Dict[str, Any]:
        """Проверяет качество ответа через GigaChat"""
        try:
            from langchain_gigachat.chat_models import GigaChat
            
            giga = GigaChat(
                credentials=os.getenv("GIGACHAT_API_KEY"),
                verify_ssl_certs=False,
                model="GigaChat:latest"
            )
            
            system_prompt = """Ты - эксперт по оценке качества ответов. 
Проанализируй ответ на вопрос и определи, является ли он полным и информативным.

Ответь в формате JSON:
{
    "needs_improvement": true/false,
    "reason": "причина, если нужно улучшение",
    "confidence": "уровень уверенности в ответе (0-1)"
}

Ответ считается неполным если:
- Содержит фразы типа "не предоставлена информация", "недостаточно данных", "неизвестно"
- Слишком общий или неконкретный
- Не отвечает на основной вопрос
- Содержит только частичную информацию"""

            user_prompt = f"""
Вопрос: {question}

Ответ для анализа: {answer}

Оцени качество этого ответа."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = giga.invoke(messages).content
            quality_assessment = json.loads(response)
            return quality_assessment
            
        except Exception as e:
            print(f"Ошибка проверки качества ответа: {e}")
            return {
                "needs_improvement": False,
                "reason": f"Ошибка проверки: {str(e)}",
                "confidence": 0.0
            }
    
    def _enhance_answer_with_rag(self, question: str, initial_answer: str, existing_results: List[Dict[str, Any]]) -> str:
        """Улучшает ответ с помощью дополнительного RAG поиска"""
        try:
            # Проверяем доступность RAG
            if not hasattr(self.rag_agent, 'vectorstore') or not self.rag_agent.vectorstore:
                print("RAG индекс недоступен для улучшения ответа")
                return initial_answer
            
            # Формируем расширенный поисковый запрос
            enhanced_query = self._build_enhanced_query(question, initial_answer, existing_results)
            print(f"Расширенный запрос для RAG: {enhanced_query}")
            
            # Выполняем RAG поиск
            rag_results = self.rag_agent.retrieve(enhanced_query, top_k=5)
            
            if not rag_results:
                print("RAG поиск не дал дополнительных результатов")
                return initial_answer
            
            # Форматируем RAG результаты
            formatted_rag_results = [{
                "step_id": "enhanced_rag",
                "status": "completed", 
                "output": rag_results,
                "summary": f"Дополнительный RAG поиск: найдено {len(rag_results)} результатов"
            }]
            
            # Повторно синтезируем ответ с учетом RAG результатов
            enhanced_answer = self.analysis_agent.synthesize_answer(
                question=question,
                table_results=existing_results,
                text_results=formatted_rag_results
            )
            
            return str(enhanced_answer)
            
        except Exception as e:
            print(f"Ошибка улучшения ответа через RAG: {e}")
            return initial_answer
    
    def _build_enhanced_query(self, question: str, initial_answer: str, existing_results: List[Dict[str, Any]]) -> str:
        """Строит расширенный поисковый запрос для RAG"""
        query_parts = [question]
        
        # Добавляем ключевые слова из исходного вопроса
        question_keywords = [word for word in question.split() if len(word) > 3]
        
        # Извлекаем сущности из существующих результатов
        entities = []
        for result in existing_results:
            output = result.get("output", [])
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, dict):
                        if "entity" in item:
                            entities.append(item["entity"])
                        if "summary" in item:
                            # Извлекаем важные слова из summary
                            summary_words = [word for word in item["summary"].split() 
                                           if len(word) > 4 and word[0].isupper()]
                            entities.extend(summary_words[:3])  # Берем первые 3
        
        # Удаляем дубликаты и добавляем к запросу
        unique_entities = list(set(entities))[:5]  # Максимум 5 сущностей
        query_parts.extend(unique_entities)
        
        return " ".join(query_parts)
    
    def _format_json_response(self, question: str, session_log: Dict[str, Any], final_answer: Any) -> Dict[str, Any]:
        """Форматирует финальный ответ в JSON согласно требованиям пользователя"""
        
        # Извлекаем функциональные вызовы из логов
        function_calls = []
        for step in session_log["steps"]:
            function_calls.append({
                "agent": step["agent"],
                "function": step["action"],
                "input": question if step["step"] == 1 else "processed_data",
                "output": self._summarize_output(step["result"]),
                "success": step["status"] in ["success", "completed"]
            })
        
        # Извлекаем reasoning из процесса
        reasoning_parts = []
        if session_log["steps"]:
            for step in session_log["steps"]:
                if "step_id" in step:
                    reasoning_parts.append(f"{step.get('step_id', step['step'])} ({step['agent']}): {step.get('description', step['action'])}")
                else:
                    reasoning_parts.append(f"Шаг {step['step']} ({step['agent']}): {step['action']}")
        
        reasoning = " → ".join(reasoning_parts)
        
        # Извлекаем прямой ответ из final_answer
        direct_answer = "Ответ не найден"
        confidence = 0.0
        
        if isinstance(final_answer, str):
            direct_answer = final_answer
            confidence = 0.8 if final_answer != "Ответ не найден" else 0.0
        
        return {
            "status": "success",
            "question": question,
            "answer": direct_answer,
            "reasoning": reasoning,
            "confidence": confidence,
            "function_calls": function_calls,
            "processing_time_seconds": session_log.get("processing_time", 0),
            "session_id": session_log["session_id"],
            "agents_involved": [step["agent"] for step in session_log["steps"]]
        }
    
    def _summarize_output(self, output: Any) -> str:
        """Создает краткую сводку результата для JSON"""
        if isinstance(output, dict):
            if "status" in output:
                return f"Status: {output['status']}"
            return str(output)[:100] + "..." if len(str(output)) > 100 else str(output)
        elif isinstance(output, list):
            return f"List with {len(output)} items"
        else:
            return str(output)[:100] + "..." if len(str(output)) > 100 else str(output)


# def main():
#     try:
#         system = MultiAgentSystem("preprocessed_data")
        
#         # Тестовые вопросы
#         test_questions = [
#             "Which battle took place in largest country in sub-Saharan Africa ?",
#         ]
        
#         for i, question in enumerate(test_questions, 1):
#             print(f"\nТЕСТ {i}/{len(test_questions)}: {question}")
#             print("-" * 50)
            
#             result = system.process_question(question, use_rag=True)
            
#             print(f"\nРЕЗУЛЬТАТ:")
#             print(json.dumps(result, ensure_ascii=False, indent=2))
#     except Exception as e:
#         print(f"Критическая ошибка: {e}")
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()