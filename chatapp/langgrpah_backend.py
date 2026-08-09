from typing import List, TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages


# Llama 3.2
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


class MessageDict(TypedDict):
    messages: Annotated[
        List[BaseMessage],
        add_messages
    ]


def chat_message(state: MessageDict):

    messages = state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# Checkpointer
checkpointer = InMemorySaver()


# Create Graph
graph = StateGraph(MessageDict)


# Add node
graph.add_node(
    "chat_message",
    chat_message
)


# Add edges
graph.add_edge(
    START,
    "chat_message"
)

graph.add_edge(
    "chat_message",
    END
)


# Compile
chatbot = graph.compile(
    checkpointer=checkpointer
)