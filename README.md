# Agentic RAG

Agentic RAG is a project designed to facilitate Retrieval-Augmented Generation (RAG) tasks using a combination of modern web technologies and AI models. This project leverages FastAPI for building APIs, and integrates various language processing libraries to enhance its capabilities.

## APIs

### Agent RAG Endpoint
- **Path**: `/agent_rag`
- **Method**: `POST`
- **Description**: Handles requests related to the RAG agent, processing payloads of type `AgentRequest`.

### Upload File Endpoint
- **Path**: `/upload`
- **Method**: `POST`
- **Description**: Allows users to upload files, accepting a file parameter of type `UploadFile`.

## Technologies Used

- **FastAPI**: For building high-performance APIs.
- **FastEmbed, LangChain, LangChain-Community, LangChain-OpenAI, LangChain-Qdrant, LangGraph, LangSmith**: Libraries for language processing and RAG tasks.
- **OpenAI**: For interacting with AI models.
- **Pydantic-Settings**: For data validation and settings management.
- **PyMongo**: For MongoDB interactions.
- **PyMuPDF**: For PDF file manipulation.
- **Python-Multipart**: For handling file uploads.
- **Qdrant-Client**: For vector search operations.
- **Secure-SMTPLib**: For secure email sending.
- **Uvicorn**: As the ASGI server.

## Getting Started

To get started with the project, ensure you have Python 3.11 or higher installed. Install the dependencies listed in `pyproject.toml` and run the application using Uvicorn.

