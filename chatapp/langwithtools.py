from langchain_core.messages import  BaseMessage , HumanMessage , AIMessage
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import Tool, DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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