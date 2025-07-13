from pathlib import Path
from typing import List, Union
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document


class DocumentLoader:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def load(self) -> List[Document]:
        all_docs = []

        # If path is a file, load that single file
        if self.path.is_file():
            loader = self._get_loader(self.path)
            if loader:
                docs = loader.load()
                for doc in docs:
                    doc.metadata["file_name"] = self.path.name
                    doc.metadata["file_path"] = str(self.path.resolve())
                all_docs.extend(docs)
            return all_docs

        # If path is a directory, load all supported files
        elif self.path.is_dir():
            for file_path in self.path.rglob("*"):
                loader = self._get_loader(file_path)
                if loader is None:
                    continue

                docs = loader.load()
                for doc in docs:
                    doc.metadata["file_name"] = file_path.name
                    doc.metadata["file_path"] = str(file_path.resolve())

                all_docs.extend(docs)
            return all_docs

        else:
            raise ValueError(f"Invalid path: {self.path}. Must be a file or directory.")

    def _get_loader(self, file_path: Path):
        if file_path.suffix.lower() == ".pdf":
            print("Trying to load PDF file:", file_path)
            return PyMuPDFLoader(str(file_path), mode="page", extract_tables="markdown")
        elif file_path.suffix.lower() == ".txt":
            print("Trying to load text file:", file_path)
            return TextLoader(str(file_path), encoding="utf-8")
        return None
