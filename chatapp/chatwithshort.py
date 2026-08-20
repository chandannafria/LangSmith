from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph , START , END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage , AIMessage ,HumanMessage

## llm model

llm = ChatOllama(
    model = "llama3.2"
)

## define state
class chat_state(TypedDict):
    messages : Annotated[List[BaseMessage], add_messages]
    
def chat_node(state:chat_state):
    message = state["messages"]
    response = llm.invoke(message)
    
    return {
        "messages" : [response]
    }
    
# checkpointer
checkpointer = InMemorySaver()


## stategraph
graph = StateGraph(chat_state)

graph.add_node("chat_node", chat_node)

graph.add_edge(START , "chat_node")
graph.add_edge("chat_node", END)

# compile

chatbot = graph.compile(checkpointer= checkpointer)


## thread id

CONFIG = {"configurable": {"thread_id":"thread_1"}}

# initial_state = {"messages":[HumanMessage(content="what is capital of punjab ")]}

# qa = chatbot.invoke(initial_state, config=CONFIG)["messages"][-1].content

# print(qa)

while True:
    user_input= input("enter text..")
    print("user:",user_input)
    
    if user_input.lower() in ["exit", "quit"]:
        break
    
    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config= CONFIG)
    print("AI:", response['messages'][-1].content)
    
    