import uuid
from datetime import datetime, timezone
from typing import Optional

def get_current_time():
    return datetime.now(timezone.utc).isoformat()


def generate_uuid():
    return str(uuid.uuid4())

def build_booking_record(
    username: str,
    email: str,
    booking_date: str,
    booking_time: Optional[str]
):
    return {
        "booking_id": generate_uuid(),
        "username": username,
        "email": email,
        "booking_date": booking_date,
        "booking_time": booking_time,
        "created_at": get_current_time(),
    }


def build_metadata_records_from_documents(documents: list):
    """
    Takes a list of LangChain Document objects (from one file)
    and returns a list of metadata records, grouped under one document_id.
    """
    document_id = generate_uuid()  # shared ID for all chunks of this document
    records = []

    for doc in documents:
        metadata = doc.metadata or {}
        record = {
            "chunk_id": generate_uuid(),
            "document_id": document_id,
            "page_content": doc.page_content,
            "metadata": {
                "file_name": metadata.get("file_name"),
                "page_no": metadata.get("page_no"),
                "chunking_strategy": metadata.get("chunking_strategy"),
                "embedding_model": metadata.get("embedding_model"),
                "created_at": get_current_time(),
            }
        }
        records.append(record)

    return records
    
    
def build_rag_message(question, message_response):
    messages = []
    records = {
        "message_id": generate_uuid(),
        "user_query": question,
        "message_response": message_response,
        "createdAt": get_current_time(),
    }
    messages.append(records)
    return messages


def build_conversation_record(
    messages,
    conversation_id: Optional[str] = None,
):
    return {
        "conversation_id": conversation_id if conversation_id else generate_uuid(),
        "messages": messages,
        "createdAt": get_current_time(),
    }