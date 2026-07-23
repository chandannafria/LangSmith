from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_ollama import OllamaLLM

load_dotenv()

llm = OllamaLLM(model="llama3.2")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [
            AIMessage(content=response)
        ]
    }


graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile()


initial_state = {
    "messages": [
        HumanMessage(content="What is the capital of India?")
    ]
}

final_state = chatbot.invoke(initial_state)

print(final_state)