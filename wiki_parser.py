import requests
from bs4 import BeautifulSoup

def fetch_wiki_intro(relative_url: str, max_sentences: int = 10) -> str:
    """
    Загружает статью по относительному wikilink и возвращает первые предложения.
    """
    base = "https://en.wikipedia.org"
    resp = requests.get(base + relative_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    paras = soup.select("p")
    intro = ""
    for p in paras:
        text = p.get_text().strip()
        if text:
            intro += text + " "
        if len(intro.split('.')) >= max_sentences:
            break
    return intro.strip()
