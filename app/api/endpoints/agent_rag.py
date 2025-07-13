from fastapi import APIRouter, Request
from services.rag_agent.agent_graph import get_graph
from utils.mongodb_message_builder import build_rag_message, build_conversation_record
from utils.crud import ConversationStore
import os
from services.rag_agent.chat_history import get_chat_history_from_mongo
from schemas.schemas import AgentRequest, AgentResponse


router = APIRouter()
graph = get_graph()


@router.post("/agent_rag", response_model=AgentResponse)
async def agent_rag_endpoint(payload: AgentRequest, request: Request):
    conversation_store = ConversationStore(
        collection_name="rag_conversations",
    )

    # Determine conversation ID
    if (
        hasattr(request.app.state, "conversation_id")
        and request.app.state.conversation_id
    ):
        conv_id = request.app.state.conversation_id
    else:
        conv_id = None

    chat_history = get_chat_history_from_mongo(conv_id)

    print(f"Chat history for conversation {conv_id}: {chat_history}")

    # Invoke agent graph with history
    result = await graph.ainvoke(
        {
            "user_input": payload.query,
            "chat_history": chat_history,
        },
        config={"run_name": "agent_rag_flow", "metadata": {"query": payload.query}},
    )

    if result.get("selected_tool") == "search_knowledge_base":
        # Save new Q&A
        messages = build_rag_message(payload.query, result["tool_output"])
        conversation = build_conversation_record(messages=messages)

        if conv_id:
            await conversation_store.append_message_to_conversation(conv_id, messages)
        else:
            conv_id = await conversation_store.create_new_conversation(conversation)
            request.app.state.conversation_id = conv_id

    return AgentResponse(result=result["tool_output"])
