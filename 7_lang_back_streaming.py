from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

llm= ChatGroq(model="openai/gpt-oss-20b")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

stream =  chatbot.stream(
    {'messages': [HumanMessage(content='what is ISA-95 architecture?')]},
    config={"configurable": {'thread_id':'thread-1'}},
    stream_mode="messages"
)

for message_chunk, metadata in stream:
    if message_chunk.content:
        print(message_chunk.content, end=" ",flush=True)
