from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from uuid import uuid4
from services.ingestion.loader import DocumentLoader
from services.ingestion.splitter import TextChunker
from services.ingestion.vectorstore import LangChainQdrantStore


router = APIRouter()

MAX_FILE_SIZE_MB = 10  # optional limit (in MB)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")
    
    # Optional: File size check
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")
    
    # Save with unique filename to avoid collision
    root_dir = Path.cwd()
    unique_name = f"{uuid4().hex}_{file.filename}"
    save_path = root_dir / unique_name

    try:
        # Save the file
        with open(save_path, "wb") as f:
            f.write(contents)

        # Load and split document
        loader = DocumentLoader(str(save_path))
        documents = loader.load()

        documents = TextChunker().split_documents(documents)
        
    #   #  Optionally store in vector DB
        vector_store = LangChainQdrantStore(collection_name="uploaded_documents")
        vector_store.store_documents(documents)

        # Clean up uploaded file
        save_path.unlink()

        return JSONResponse(content={
            "filename": file.filename,
            "num_documents": len(documents),
            "preview": documents[0].page_content[:300] if documents else "",
            "metadata": documents[0].metadata if documents else {},
            "collection_name": vector_store.collection_name
        })

    except Exception as e:
        # Cleanup on error
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))
