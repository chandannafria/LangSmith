from langchain_core.messages import  BaseMessage , HumanMessage , AIMessage
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import Tool, DuckDuckGoSearchRun, WikipediaQueryRun, wikipedia
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool


llm = OllamaLLM(
    model="llama3.2")

prompt = ChatPromptTemplate.from_messages(
    [ ("system",  "you are a helpful AI...."),
     ("human", "explain {topic}")
     
     
     ]
)

chain = prompt | llm | StrOutputParser()

# invoke the runnable chain with input
chain_output = chain.invoke({"topic": "capital of india"})


search = DuckDuckGoSearchRun()
@tool
def duckduckgo_search(query: str) -> str:
    """Search the web using DuckDuckGo and return relevant results."""
    return search.run(query)
    

# @tool
# def wikipedia_search(query: str) -> str:
#     """Search Wikipedia for factual information about a topic."""
    
#     try:
#         return wikipedia.summary(query, sentences=3)
#     except Exception as e:
#         return f"Wikipedia search failed: {e}"
    
tools = [
    duckduckgo_search,
    # wikipedia_search
]

# llm_with_tool  = llm.bind(tools)


print (chain_output)

print(duckduckgo_search.name)
print(duckduckgo_search.description)
print(duckduckgo_search.args)