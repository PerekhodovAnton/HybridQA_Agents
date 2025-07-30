import os
import json
import glob
from tqdm import tqdm
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
import time
from datetime import datetime
from src.wiki_parser import fetch_wiki_intro
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def estimate_tokens(text: str) -> int:
    """
    Приблизительно оценивает количество токенов в тексте
    Примерно 1 токен = 4 символа для русского/английского текста
    """
    return len(text) // 4


def truncate_text_for_embeddings(text: str, max_tokens: int = 450) -> str:
    """
    Обрезает текст до максимального количества токенов для embeddings
    
    Args:
        text: Исходный текст
        max_tokens: Максимальное количество токенов (по умолчанию 500, с запасом от лимита 514)
        
    Returns:
        Обрезанный текст
    """
    if estimate_tokens(text) <= max_tokens:
        return text
    
    # Приблизительно вычисляем максимальную длину в символах
    max_chars = max_tokens * 4
    
    if len(text) <= max_chars:
        return text
    
    # Обрезаем по словам, а не по символам
    words = text.split()
    truncated = ""
    
    for word in words:
        test_text = truncated + " " + word if truncated else word
        if estimate_tokens(test_text) > max_tokens:
            break
        truncated = test_text
    
    return truncated + "..." if truncated != text else truncated


def create_chunked_documents(text: str, metadata: dict, chunk_size: int = 400) -> List[Document]:
    """
    Создает несколько документов из длинного текста, разбивая его на части
    
    Args:
        text: Длинный текст
        metadata: Метаданные для документов
        chunk_size: Размер куска в токенах
        
    Returns:
        Список документов
    """
    documents = []
    
    if estimate_tokens(text) <= chunk_size:
        # Если текст короткий, создаем один документ
        truncated_text = truncate_text_for_embeddings(text, chunk_size)
        documents.append(Document(page_content=truncated_text, metadata=metadata))
    else:
        # Разбиваем длинный текст на части
        words = text.split()
        current_chunk = ""
        chunk_num = 0
        
        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word
            
            if estimate_tokens(test_chunk) > chunk_size:
                # Сохраняем текущий кусок
                if current_chunk:
                    chunk_metadata = metadata.copy()
                    chunk_metadata['chunk_id'] = chunk_num
                    chunk_metadata['is_chunked'] = True
                    
                    documents.append(Document(
                        page_content=current_chunk,
                        metadata=chunk_metadata
                    ))
                    chunk_num += 1
                
                current_chunk = word
            else:
                current_chunk = test_chunk
        
        # Добавляем последний кусок
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_id'] = chunk_num
            chunk_metadata['is_chunked'] = True
            
            documents.append(Document(
                page_content=current_chunk,
                metadata=chunk_metadata
            ))
    
    return documents


class RetrievalAgent:
    
    def __init__(self, json_folder: str, cache_dir: str = "cached_articles"):
        self.json_folder = json_folder
        self.cache_dir = cache_dir
        self.api_key = os.getenv("GIGACHAT_API_KEY")
        self.embedder = GigaChatEmbeddings(
            credentials=self.api_key,
            verify_ssl_certs=False
        )
        self.docs: List[Document] = []
        self.vectorstore = None
        self.wiki_cache: Dict[str, str] = {}  # В памяти кэш
        
        # Создаем директорию для кэша если не существует
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_filename(self, wiki_url: str) -> str:
        """Генерирует имя файла для кэша статьи"""
        # Создаем безопасное имя файла из URL
        safe_name = wiki_url.replace("/wiki/", "").replace("/", "_").replace(".", "_")
        return os.path.join(self.cache_dir, f"wiki_{safe_name}.json")
    
    def _save_article_to_cache(self, wiki_url: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        try:
            cache_file = self._get_cache_filename(wiki_url)
            
            cache_data = {
                "wiki_url": wiki_url,
                "content": content,
                "cached_at": datetime.now().isoformat(),
                "content_length": len(content),
                "estimated_tokens": estimate_tokens(content),
                "metadata": metadata or {}
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            print(f"Сохранено в кэш: {wiki_url} ({estimate_tokens(content)} токенов)")
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения в кэш {wiki_url}: {e}")
            return False
    
    def _load_article_from_cache(self, wiki_url: str) -> Optional[Dict[str, Any]]:
        try:
            cache_file = self._get_cache_filename(wiki_url)
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                tokens = cache_data.get('estimated_tokens', estimate_tokens(cache_data.get('content', '')))
                print(f"Загружено из кэша: {wiki_url} ({tokens} токенов)")
                return cache_data
            
            return None
            
        except Exception as e:
            print(f"Ошибка загрузки из кэша {wiki_url}: {e}")
            return None
    
    def _fetch_or_load_article(self, wiki_url: str, max_sentences: int = 15) -> str:
        # Сначала пытаемся загрузить из кэша
        cached_data = self._load_article_from_cache(wiki_url)
        
        if cached_data and cached_data.get('content'):
            return cached_data['content']
        
        # Если в кэше нет, скачиваем
        print(f"Скачиваем: {wiki_url}")
        content = fetch_wiki_intro(wiki_url, max_sentences)
        
        if content:
            self._save_article_to_cache(wiki_url, content, {
                "max_sentences": max_sentences,
                "source": "wikipedia_download"
            })
        
        return content
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        cache_files = glob.glob(os.path.join(self.cache_dir, "wiki_*.json"))
        
        total_size = 0
        total_articles = len(cache_files)
        total_tokens = 0
        cache_info = []
        
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    file_size = os.path.getsize(cache_file)
                    tokens = data.get('estimated_tokens', estimate_tokens(data.get('content', '')))
                    
                    total_size += file_size
                    total_tokens += tokens
                    
                    cache_info.append({
                        "url": data.get('wiki_url', 'unknown'),
                        "cached_at": data.get('cached_at', 'unknown'),
                        "content_length": data.get('content_length', 0),
                        "estimated_tokens": tokens,
                        "file_size": file_size
                    })
            except Exception as e:
                print(f"Ошибка чтения {cache_file}: {e}")
        
        return {
            "total_articles": total_articles,
            "total_cache_size_bytes": total_size,
            "total_cache_size_mb": round(total_size / 1024 / 1024, 2),
            "total_estimated_tokens": total_tokens,
            "avg_tokens_per_article": round(total_tokens / max(total_articles, 1), 1),
            "cache_directory": self.cache_dir,
            "articles": cache_info[:10]  # Показываем первые 10 для примера
        }
    
    def clear_cache(self, confirm: bool = False) -> bool:
        if not confirm:
            print("Для очистки кэша вызовите clear_cache(confirm=True)")
            return False
        
        try:
            cache_files = glob.glob(os.path.join(self.cache_dir, "wiki_*.json"))
            
            for cache_file in cache_files:
                os.remove(cache_file)
            
            print(f"Очищено {len(cache_files)} файлов из кэша")
            return True
            
        except Exception as e:
            print(f"Ошибка очистки кэша: {e}")
            return False

    def extract_unique_wiki_links(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Извлекает все уникальные Wikipedia ссылки из JSON файлов
        
        Returns:
            Словарь {wiki_url: [список метаданных для этой ссылки]}
        """
        unique_links = defaultdict(list)
        
        for json_path in glob.glob(os.path.join(self.json_folder, "*.json")):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Обрабатываем каждую запись
                for record in tqdm(data, desc=f"Записи из {os.path.basename(json_path)}"):
                    question_id = record.get('question_id', 'unknown')
                    question = record.get('question', '')
                    table_id = record.get('table_id', '')
                    
                    # Извлекаем ссылки из nodes
                    nodes = record.get('nodes', [])
                    for node in nodes:
                        if len(node) >= 7:  # Проверяем корректность структуры
                            entity, position, wiki_link, summary, score, role, method = node[:7]
                            
                            if wiki_link and wiki_link.startswith('/wiki/'):
                                # Сохраняем метаданные для каждой ссылки
                                unique_links[wiki_link].append({
                                    'entity': entity,
                                    'position': position,
                                    'summary': summary,
                                    'score': score,
                                    'role': role,
                                    'method': method,
                                    'question_id': question_id,
                                    'question': question,
                                    'table_id': table_id
                                })
                                
            except Exception as e:
                print(f"Ошибка при обработке файла {json_path}: {e}")
                continue
                
        print(f"Найдено {len(unique_links)} уникальных Wikipedia статей")
        return dict(unique_links)
    
    def build_enhanced_index(self) -> FAISS:
        unique_links = self.extract_unique_wiki_links()
        
        if not unique_links:
            print("Не найдено Wikipedia ссылок для индексирования")
            return None
            
        print("Загружаем полные тексты Wikipedia статей (с кэшированием)...")
        
        failed_embeddings = 0
        successful_embeddings = 0
        
        for wiki_url, metadata_list in tqdm(unique_links.items(), desc="Загрузка статей"):
            try:
                # Используем новую функцию с кэшированием
                full_text = self._fetch_or_load_article(wiki_url, max_sentences=10)  # Уменьшили с 20 до 10
                
                if full_text and len(full_text) > 50:
                    self.wiki_cache[wiki_url] = full_text
                    
                    # Создаем документ для каждой уникальной комбинации метаданных
                    for metadata in metadata_list:
                        # Комбинируем полный текст с оригинальным summary
                        combined_text = f"""Entity: {metadata['entity']}
                                        Role: {metadata['role']}
                                        Summary: {metadata['summary']}

                                        Article: {full_text}"""
                        
                        # Проверяем размер перед созданием документов
                        estimated_tokens = estimate_tokens(combined_text)
                        
                        if estimated_tokens > 500:
                            print(f"Текст слишком длинный ({estimated_tokens} токенов), разбиваем на части: {metadata['entity']}")
                            
                            # Создаем несколько документов из длинного текста
                            docs = create_chunked_documents(combined_text, {
                                'type': 'wikipedia_article',
                                'entity': metadata['entity'],
                                'wiki_url': wiki_url,
                                'role': metadata['role'],
                                'method': metadata['method'],
                                'score': metadata['score'],
                                'question_id': metadata['question_id'],
                                'question': metadata['question'],
                                'table_id': metadata['table_id'],
                                'original_summary': metadata['summary'],
                                'full_text_length': len(full_text)
                            })
                            
                            self.docs.extend(docs)
                            successful_embeddings += len(docs)
                        else:
                            # Обрезаем текст для безопасности
                            safe_text = truncate_text_for_embeddings(combined_text, 500)
                            
                            doc = Document(
                                page_content=safe_text,
                                metadata={
                                    'type': 'wikipedia_article',
                                    'entity': metadata['entity'],
                                    'wiki_url': wiki_url,
                                    'role': metadata['role'],
                                    'method': metadata['method'],
                                    'score': metadata['score'],
                                    'question_id': metadata['question_id'],
                                    'question': metadata['question'],
                                    'table_id': metadata['table_id'],
                                    'original_summary': metadata['summary'],
                                    'full_text_length': len(full_text),
                                    'estimated_tokens': estimate_tokens(safe_text)
                                }
                            )
                            
                            self.docs.append(doc)
                            successful_embeddings += 1
                        
                # Небольшая задержка только для новых загрузок (не для кэша)
                if not self._load_article_from_cache(wiki_url):
                    time.sleep(0.1)
                
            except Exception as e:
                print(f"Ошибка при обработке {wiki_url}: {e}")
                failed_embeddings += 1
                continue
        
        print(f"Создано {len(self.docs)} документов для индексирования")
        print(f"Успешно: {successful_embeddings}, Ошибок: {failed_embeddings}")
        
        # Показываем статистику по размерам
        token_stats = [estimate_tokens(doc.page_content) for doc in self.docs]
        if token_stats:
            print(f"Размеры документов (токены): мин={min(token_stats)}, макс={max(token_stats)}, средн={sum(token_stats)//len(token_stats)}")
        
        # Шаг 3: Создаем FAISS индекс
        if not self.docs:
            print("Нет документов для создания индекса")
            return None
            
        try:
            # Создаем индекс пакетами для лучшей производительности
            batch_size = 5  # Уменьшили с 10 до 5 для безопасности
            faiss_index = None
            
            with tqdm(total=len(self.docs), desc="Индексирование документов") as pbar:
                for i in range(0, len(self.docs), batch_size):
                    batch = self.docs[i:i + batch_size]
                    
                    try:
                        if faiss_index is None:
                            # Создаем индекс с первым батчем
                            faiss_index = FAISS.from_documents(batch, self.embedder)
                        else:
                            # Добавляем остальные батчи
                            faiss_index.add_documents(batch)
                        
                        pbar.update(len(batch))
                        
                    except Exception as batch_error:
                        print(f"Ошибка в батче {i//batch_size + 1}: {batch_error}")
                        # Пропускаем проблемный батч
                        pbar.update(len(batch))
                        continue
            
            self.vectorstore = faiss_index
            print("Индекс успешно создан!")
            return faiss_index
            
        except Exception as e:
            print(f"Ошибка при создании индекса: {e}")
            return None
    
    def retrieve(self, query: str, top_k: int = 5, filter_by_role: str = None) -> List[Dict[str, Any]]:
        if not self.vectorstore:
            raise ValueError("Индекс не создан - запустите build_enhanced_index() сначала")
        
        # Выполняем поиск
        results = self.vectorstore.similarity_search(query, k=top_k * 2)  # Берем больше для фильтрации
        
        # Применяем фильтры если нужно
        filtered_results = []
        for doc in results:
            if filter_by_role and doc.metadata.get('role', '').lower() != filter_by_role.lower():
                continue
            
            filtered_results.append({
                'text': doc.page_content,
                'metadata': doc.metadata,
                'entity': doc.metadata.get('entity', 'Unknown'),
                'role': doc.metadata.get('role', 'Unknown'),
                'wiki_url': doc.metadata.get('wiki_url', ''),
                'relevance_score': doc.metadata.get('score', 0)
            })
            
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results
    
# Создание агента
# agent = RetrievalAgent("preprocessed_data")
# faiss_store = agent.build_enhanced_index()
# agent.vectorstore.save_local("my_index")

# test
# agent.vectorstore = FAISS.load_local("my_index", agent.embedder, allow_dangerous_deserialization=True)
# results = agent.retrieve("What is the population of the hometown of the 2012 Gatorade Player of the Year ?", top_k=3)
# print(results)

