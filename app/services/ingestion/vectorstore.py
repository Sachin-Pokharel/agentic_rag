from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from qdrant_client import QdrantClient, AsyncQdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from typing import List
import uuid
import os
from dotenv import load_dotenv
from utils.mongodb_message_builder import build_metadata_records_from_documents
from utils.utils import clean_page_content
from utils.crud import ConversationStore


load_dotenv()
# Singleton instance
vector_store_singleton = None


class LangChainQdrantStore:
    def __init__(self, collection_name: str, embedding_model_name: str = "text-embedding-3-small"):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        # Set up Qdrant clients (remote)
        self.client = QdrantClient(
            url=os.getenv("QDRANT_HOST_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.aclient = AsyncQdrantClient(
            url=os.getenv("QDRANT_HOST_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        # Embedding models
        self.embedding_model = OpenAIEmbeddings(
            model=self.embedding_model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.sparse_model = FastEmbedSparse(model_name="Qdrant/bm25")

        # Ensure collection exists
        self._create_collection_if_not_exists()

        # Create LangChain-compatible vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            sparse_embedding=self.sparse_model,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

    def _create_collection_if_not_exists(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=1536, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                }
            )

    def store_documents(self, documents: List[Document]) -> List[str]:
        """
        Stores LangChain Document chunks in the vector store.
        """
        ids = [str(uuid.uuid4()) for _ in documents]
        for doc in documents:
            doc.page_content = clean_page_content(doc.page_content)  # Clean page content
            doc.metadata = {
                **doc.metadata,
                "embedding_model": self.embedding_model_name,
            }
            
        conversation_store = ConversationStore(collection_name=os.getenv('MONGODB_RAG_UPLOAD_METADATA_COLLECTION'))
        conversation_store.get_collection().insert_many(
            build_metadata_records_from_documents(documents)
        )
        
        return self.vector_store.add_documents(documents, ids=ids)

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Performs a hybrid similarity search.
        """
        return self.vector_store.similarity_search(query=query, k=k)

    def search_with_scores(self, query: str, k: int = 5):
        return self.vector_store.similarity_search_with_score(query=query, k=k)
