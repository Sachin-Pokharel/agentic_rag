import os
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import PromptTemplate
from schemas.schemas import AgentState, AgentAction
from .tools import search_knowledge_base, book_interview
from utils.crud import ConversationStore
from utils.mongodb_message_builder import build_booking_record
from dotenv import load_dotenv

load_dotenv()

# LangSmith setup
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "agentic_rag"

# Tool mapping
TOOL_MAP = {
    "search_knowledge_base": search_knowledge_base,
    "book_interview": book_interview
}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv('OPENAI_API_KEY'))
synth_llm = OpenAI(model="gpt-4o-mini", temperature=0)


async def process_user_input(state: AgentState):
    structured_llm = llm.with_structured_output(AgentAction)
    prompt = PromptTemplate.from_template("""
    You are an intelligent agent that can perform two types of actions:

    1. search_knowledge_base: for general queries, questions, and searching for information
    2. book_interview: for scheduling interviews with receiver_email, user_name, date/time

    Here is the chat history so far (if any):

    {history}

    Now analyze the latest query:
    User Query: {input}

    Respond with:
    - The tool name
    - Reasoning
    - The structured action
    """)

    try:
        chat_history_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in state.get("chat_history", []))
        result = await structured_llm.ainvoke(prompt.format(input=state["user_input"], history=chat_history_text), config={"run_name": "process_user_input"})
        state["selected_tool"] = result.tool_name

        if result.action.action_type == "search":
            state["tool_input"] = {"query": result.action.query}
        elif result.action.action_type == "booking":
            state["tool_input"] = {
                "receiver_email": result.action.receiver_email,
                "user_name": result.action.user_name,
                "appointment_date": result.action.appointment_date,
                "appointment_time": result.action.appointment_time
            }
            booking_store = ConversationStore(collection_name=os.getenv('MONGODB_BOOKING_COLLECTION'))
            booking_store.save_booking(
                build_booking_record(
                    username=result.action.user_name,
                    email=result.action.receiver_email,
                    booking_date=result.action.appointment_date,
                    booking_time=result.action.appointment_time
                )
            )

        print(f"Selected tool: {result.tool_name} - {result.reasoning}")
        print(f"Extracted input: {state['tool_input']}")

    except Exception as e:
        # fallback to search
        state["selected_tool"] = "search_knowledge_base"
        state["tool_input"] = {"query": state["user_input"]}
        print(f"Fallback to search due to error: {str(e)}")

    return state

def run_tool(state: AgentState):
    print(f"Running tool: {state['selected_tool']} with input: {state['tool_input']}")
    tool_func = TOOL_MAP.get(state["selected_tool"])
    if tool_func:
        try:
            result = tool_func.invoke(state["tool_input"])
            state["tool_output"] = result
        except Exception as e:
            state["tool_output"] = f"Error executing tool: {str(e)}"
    else:
        state["tool_output"] = f"Unknown tool: {state['selected_tool']}"
    return state

async def synthesize_search_results(state: AgentState):
    if state['selected_tool'] == "search_knowledge_base":
        docs = state.get("tool_output", [])
        if not docs:
            state["tool_output"] = "No relevant information found in the knowledge base."
            return state

        # docs is expected to be List[Document]
        combined_text = "\n\n".join([doc.page_content for doc in docs])
        query = state.get("user_input", "")

        prompt = f"""
You are an expert assistant. Using the following documents, answer the query concisely and clearly.
Make sure to provide a complete and grammatically correct response.


Query: {query}

Documents:
{combined_text}

Answer:
"""

        answer = await synth_llm.ainvoke(prompt, max_tokens=2048, config={"run_name": "synthesize_search_results"})
        state["tool_output"] = answer
    return state

def postprocess_tool_output(state: AgentState):
    tool = state.get("selected_tool")
    raw_output = state.get("tool_output", "")

    if tool == "book_interview":
        if "Failed" in raw_output or "Error" in raw_output:
            state["tool_output"] = f"Sorry, we couldn't send the confirmation email."
        else:
            state["tool_output"] = f"Success! {raw_output} We look forward to your interview."
    # You can add more tool-specific postprocessing here if needed
    return state

def get_graph():
    builder = StateGraph(AgentState)
    builder.add_node("process_input", process_user_input)
    builder.add_node("run_tool", run_tool)
    builder.add_node("synthesize", synthesize_search_results)
    builder.add_node("postprocess", postprocess_tool_output)

    builder.set_entry_point("process_input")
    builder.add_edge("process_input", "run_tool")
    builder.add_edge("run_tool", "synthesize")
    builder.add_edge("synthesize", "postprocess")
    builder.set_finish_point("postprocess")

    return builder.compile()
