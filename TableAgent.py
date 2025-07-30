import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import json
import os
import glob
from wiki_parser import fetch_wiki_intro



class TableToolAgent:
    def __init__(self, cache_dir: str = "cached_articles"):
        self.cache_dir = cache_dir
        # Создаем директорию для кэша если не существует  
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_filename(self, wiki_url: str) -> str:
        """Генерирует имя файла для кэша статьи"""
        safe_name = wiki_url.replace("/wiki/", "").replace("/", "_").replace(".", "_")
        return os.path.join(self.cache_dir, f"wiki_{safe_name}.json")
    
    def _load_cached_article(self, wiki_url: str) -> Optional[str]:
        """
        Загружает статью из локального кэша
        
        Args:
            wiki_url: URL Wikipedia статьи (например, "/wiki/Cornwall,_New_York")
            
        Returns:
            Содержимое статьи из кэша или None если не найдено
        """
        try:
            cache_file = self._get_cache_filename(wiki_url)
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                content = cache_data.get('content', '')
                if content:
                    return content
            
            print(f"Статья не найдена в кэше: {wiki_url}")
            return None
            
        except Exception as e:
            print(f"Ошибка загрузки из кэша {wiki_url}: {e}")
            return None
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику кэша для отладки"""
        cache_files = glob.glob(os.path.join(self.cache_dir, "wiki_*.json"))
        return {
            "total_cached_articles": len(cache_files),
            "cache_directory": self.cache_dir,
            "cache_exists": os.path.exists(self.cache_dir)
        }

    def execute(self,
                step: Dict[str, Any],
                question: str,
                table_data: Dict[str, Any],
                nodes: List[List[Any]]
               ) -> Dict[str, Any]:
        """
        Выполняет шаг из плана рассуждения, работая с табличными данными.
        Теперь использует локально кэшированные Wikipedia статьи.

        Args:
            step: Шаг из плана рассуждения.
            question: Исходный вопрос пользователя.
            table_data: Данные таблицы, включая идентификатор и содержимое.
            nodes: Список узлов, представляющих сущности в таблице.

        Returns:
            Результаты выполнения шага с найденными сущностями и дополнительной информацией.
        """
        step_id = step.get("id")
        description = step.get("description", "")
        inputs = step.get("input") or []
        if isinstance(inputs, str):
            inputs = [inputs]

        results = []
        cache_hits = 0
        cache_misses = 0
        
        # Поиск релевантных узлов
        for entity, pos, wiki_url, summary, score, role, method in nodes:
            if any(keyword in description.lower() for keyword in (entity.lower(), role.lower())):
                result_entry = {
                    "entity": entity, 
                    "wiki": wiki_url,
                    "summary": summary, 
                    "score": score, 
                    "role": role
                }
                
                # Пытаемся загрузить статью из кэша
                if wiki_url:
                    cached_content = self._load_cached_article(wiki_url)
                    if cached_content:
                        result_entry["wiki_intro"] = cached_content
                        result_entry["source"] = "cached"
                        cache_hits += 1
                    else:
                        result_entry["wiki_error"] = "Article not found in cache"
                        result_entry["source"] = "cache_miss"
                        cache_misses += 1
                else:
                    result_entry["wiki_error"] = "No wiki URL provided"
                    result_entry["source"] = "no_url"
                
                results.append(result_entry)
        
        # Альтернативная логика поиска если основная не дала результатов
        if not results:
            for entity, pos, wiki_url, summary, score, role, method in nodes:
                if "population" in summary.lower():
                    result_entry = {
                        "entity": entity,
                        "wiki": wiki_url, 
                        "summary": summary,
                        "score": score,
                        "role": role
                    }
                    
                    if wiki_url:
                        cached_content = self._load_cached_article(wiki_url)
                        if cached_content:
                            result_entry["wiki_intro"] = cached_content
                            result_entry["source"] = "cached"
                            cache_hits += 1
                        else:
                            result_entry["wiki_error"] = "Article not found in cache"
                            result_entry["source"] = "cache_miss"
                            cache_misses += 1
                    
                    results.append(result_entry)
        
        # Формируем детальную сводку
        summary_parts = [f"Найдено {len(results)} релевантных узлов по '{description}'"]
        if cache_hits > 0:
            summary_parts.append(f"Загружено из кэша: {cache_hits}")
        if cache_misses > 0:
            summary_parts.append(f"Не найдено в кэше: {cache_misses}")
        
        return {
            "step_id": step_id,
            "status": "completed",
            "output": results,
            "summary": ". ".join(summary_parts) + ".",
            "cache_statistics": {
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "total_processed": len(results)
            }
        }

def load_json_data(file_path):
    """Загружает данные из JSON файла."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)