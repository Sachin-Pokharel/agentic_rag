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

    raw_messages = record["messages"]
    all_msgs = []
    for item in raw_messages:
        if isinstance(item, dict):
            all_msgs.append(item)
        elif isinstance(item, list):
            all_msgs.extend([m for m in item if isinstance(m, dict)])

    total_turns = len(all_msgs)
    chat_history = []

    if total_turns > summary_threshold:
        older = all_msgs[:-max_turns]
        text_to_summarize = "\n".join(
            f"User: {m.get('user_query', '')}\nAssistant: {m.get('message_response', '')}"
            for m in older
        )
        summary = synth_llm.invoke((
            "Summarize this conversation briefly, capturing key decisions and user goals:\n"
            + text_to_summarize
        )).strip()
        chat_history.append({"role": "system", "content": summary})

    recent = all_msgs[-max_turns:]
    for m in recent:
        user_q = m.get("user_query")
        assistant_a = m.get("message_response")
        if user_q:
            chat_history.append({"role": "user", "content": user_q})
        if assistant_a:
            chat_history.append({"role": "assistant", "content": assistant_a})

    return chat_history
