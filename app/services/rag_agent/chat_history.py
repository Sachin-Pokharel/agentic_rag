from utils.crud import ConversationStore
import os
from services.rag_agent.agent_graph import synth_llm

def get_chat_history_from_mongo(conversation_id: str, max_turns: int = 5, summary_threshold: int = 20):
    if not conversation_id:
        return []

    store = ConversationStore(collection_name="rag_conversations")
    record = store.find_conversation_by_id(conversation_id)
    if not record or "messages" not in record:
        return []

    all_msgs = record["messages"]
    total_turns = len(all_msgs)
    chat_history = []

    # Summarize older messages if above threshold
    if total_turns > summary_threshold:
        older = all_msgs[:-max_turns]
        text_to_summarize = "\n".join(
            f"User: {m['user_query']}\nAssistant: {m['message_response']}"
            for m in older
        )
        summary = synth_llm.invoke((
            "Summarize this conversation briefly, capturing key decisions and user goals:\n"
            + text_to_summarize
        )).strip()
        chat_history.append({"role": "system", "content": summary})

    # Add a sliding window of recent turns
    recent = all_msgs[-max_turns:]
    for m in recent:
        chat_history.append({"role": "user", "content": m["user_query"]})
        chat_history.append({"role": "assistant", "content": m["message_response"]})

    return chat_history
