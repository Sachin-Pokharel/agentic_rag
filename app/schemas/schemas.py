from pydantic import BaseModel, Field
from typing import Optional, Union, Literal, TypedDict


class AgentRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    result: str

class SearchAction(BaseModel):
    action_type: Literal["search"] = "search"
    query: str = Field(description="The search query or question to look up")

class BookingAction(BaseModel):
    action_type: Literal["booking"] = "booking"
    receiver_email: str = Field(description="Email address of the person to book interview with")
    user_name: str = Field(description="Name of the user booking the interview")
    appointment_date: str = Field(description="Date for the appointment (e.g., 'July 20', '2024-07-20')")
    appointment_time: Optional[str] = Field(default=None, description="Time for the appointment (e.g., '2 PM', '14:00')")

class AgentAction(BaseModel):
    tool_name: Literal["search_knowledge_base", "book_interview"] = Field(description="The name of the tool to use")
    reasoning: str = Field(description="Brief explanation of why this tool was chosen")
    action: Union[SearchAction, BookingAction] = Field(description="The specific action to perform with the tool")

class AgentState(TypedDict):
    user_input: str
    selected_tool: Optional[str]
    tool_input: Optional[dict]
    tool_output: Optional[Union[str, list]]
    chat_history: Optional[list]


# Unified structured output models
class SearchAction(BaseModel):
    action_type: Literal["search"] = "search"
    query: str = Field(description="The search query or question to look up")

class BookingAction(BaseModel):
    action_type: Literal["booking"] = "booking"
    receiver_email: str = Field(description="Email address of the person to book interview with")
    user_name: str = Field(description="Name of the user booking the interview")
    appointment_date: str = Field(description="Date for the appointment (e.g., 'July 20', '2024-07-20')")
    appointment_time: Optional[str] = Field(default=None, description="Time for the appointment (e.g., '2 PM', '14:00')")

class AgentAction(BaseModel):
    tool_name: Literal["search_knowledge_base", "book_interview"] = Field(description="The name of the tool to use")
    reasoning: str = Field(description="Brief explanation of why this tool was chosen")
    action: Union[SearchAction, BookingAction] = Field(description="The specific action to perform with the tool")