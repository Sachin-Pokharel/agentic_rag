from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

class TextChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
        chunking_strategy: str = "recursive",
    ):
        """
        chunk_size: max characters per chunk
        chunk_overlap: overlap characters between chunks
        separators: splitting hierarchy; defaults to ["\n\n", "\n", " ", ""]
        """
        
        self.chunking_strategy = chunking_strategy
        
        if self.chunking_strategy == "recursive":
            if separators is None:
                separators = ["\n\n", "\n", " ", ""]
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                length_function=len,
                is_separator_regex=False,
            )

    def split(self, text: str) -> List[str]:
        """Return list of text chunks"""
        return self.splitter.split_text(text)

    def split_documents(self, documents):
        """
        Wrap chunks into LangChain Document objects
        (useful for embedding or storing context).
        """
        chunked_docs = self.splitter.split_documents(documents)
        for chunk in chunked_docs:
            orig_metadata = chunk.metadata or {}
            chunk.metadata = {
                "file_name": orig_metadata.get("file_name", ""),
                "chunking_strategy": self.chunking_strategy,
                "page_no": orig_metadata.get("page", 0),            }
        return chunked_docs
        