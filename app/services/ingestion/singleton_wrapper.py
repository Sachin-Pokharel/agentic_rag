# utils/vector_store_registry.py
from services.ingestion.vectorstore import LangChainQdrantStore

_vector_store_cache = {}

def get_vector_store(collection_name: str) -> LangChainQdrantStore:
    """
    Returns a singleton instance of LangChainQdrantStore for a given collection.
    """
    if collection_name not in _vector_store_cache:
        _vector_store_cache[collection_name] = LangChainQdrantStore(collection_name)
    return _vector_store_cache[collection_name]
