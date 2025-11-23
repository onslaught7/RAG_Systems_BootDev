from nltk.stem import PorterStemmer
from functools import lru_cache
from typing import List
import string


stemmer = PorterStemmer()


@lru_cache(maxsize=1)
def _load_stop_words() -> set:
    """Load stop words from a predefioned txt file and cache the result."""
    try:
        with open("./data/stopwords.txt", "r") as f:
            stop_words = f.read().splitlines()
        return set(stop_words)
    except FileNotFoundError:
        print("Error: ./data/stopwords.txt not found.")
        return set()


def _normalize_text(text: str) -> List[str]:
    """
    Noramlize text for keyword search. Perfroms the following operations:

    - Case in-sensitivity
    - Punctuation removal
    - Tokenization
    - Stop words removal
    - Stemming    
    """
    if not isinstance(text, str):
        return []
    translator = str.maketrans("", "", string.punctuation) 
    cleaned = text.lower().translate(translator).split()

    stop_words = _load_stop_words()
    # Remove stop words
    cleaned = [word for word in cleaned if word not in stop_words]
    # stemming
    cleaned_stemmed = [stemmer.stem(word) for word in cleaned]

    return cleaned_stemmed