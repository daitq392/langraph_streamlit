import streamlit as st
import uuid
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pymongo import MongoClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from typing import Annotated

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from datetime import date, datetime
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph, START
import shutil


# MongoDB configuration
DB_NAME = "appointments_db"
COLLECTION_NAME = "appointments"

# Create a MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
appointments_collection = db[COLLECTION_NAME]
users_collection = db.users

# Define tools
@tool
def fetch_user_info(config: RunnableConfig) -> dict:
    """
    Fetch user information from the database.
    """
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        return {"error": "User ID is missing in config."}

    user = users_collection.find_one({"user_id": passenger_id})
    if not user:
        return {"error": "User not found."}

    if "_id" in user:
        user["_id"] = str(user["_id"])

    return user

@tool
def set_appointment(hospital_name: str, appointment_date: str, config: RunnableConfig) -> str:
    """
    Set an appointment in MongoDB.
    """
    user_id = config.get("configurable", {}).get("passenger_id", None)
    if not user_id:
        return "Error: User ID is missing in config."

    result = appointments_collection.update_one(
        {"user_id": user_id},
        {"$set": {"hospital_name": hospital_name, "appointment_date": appointment_date}},
        upsert=True
    )

    if result.matched_count > 0:
        return "Appointment updated successfully."
    elif result.upserted_id:
        return "New appointment created successfully."
    else:
        return "Failed to update the appointment."

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define the assistant prompt
assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant, helping users schedule appointments at hospitals."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

# Bind tools to the LLM
part_1_tools = [fetch_user_info, set_appointment]
part_1_assistant_runnable = assistant_prompt | llm.bind_tools(part_1_tools)

# Define the graph
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

builder = StateGraph(State)

def user_info(state: State):
    return {"user_info": fetch_user_info.invoke({})}

builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
builder.add_edge("fetch_user_info", "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
part_3_graph = builder.compile(checkpointer=memory)

# Streamlit app
st.title("Hospital Appointment Chatbot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize the conversation
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "passenger_id": "3442 587242",  # Hardcoded user ID for testing
            "thread_id": thread_id,
        }
    }

    # Stream the conversation
    events = part_3_graph.stream(
        {"messages": [("user", prompt)]}, config, stream_mode="values"
    )
    for event in events:
        for message in event.get("messages", []):
            if message.id not in st.session_state:
                st.session_state.messages.append({"role": "assistant", "content": message.content})
                with st.chat_message("assistant"):
                    st.markdown(message.content)