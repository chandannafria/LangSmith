from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings  import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig


# FILE UPLOAD 
file = PyPDFLoader("dl-curriculum.pdf")

docs = file.load()
# print(docs)

## text splitter
text_split = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap = 200

)
chunks = text_split.split_documents(docs)

# print(len(chunks))
# print(chunks[0])


# embedding
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# vectorstore db 

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="vectordb"
)
# print("vector db successfully")

retriever = vector_db.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":2}
)



# query = "what is AlexNet"

# result = retriever.invoke()

# for i ,  doc in enumerate(result):
#     print(f"----result---{i+1}")
#     print(doc.page_content)

# Prompt Template

prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF question-answering assistant.

Answer the question using ONLY the provided context.

If the answer is not present in the context,
say:

"I don't know based on the provided PDF."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
""")

## ollama model

llm = ChatOllama(
    model= "llama3.2"
)

## langgraph state
class messagedict (TypedDict):
    messages:Annotated[List[BaseMessage], add_messages]
    
## chat node

def chat_message(state:messagedict):
    
    #  get latest question
    question = state["messages"][-1].content
    print("/nQuestion:",question)
    
    docs = retriever.invoke(question)
    
    for i , doc in enumerate(docs):
        print(f"---chunk---{i+1}")
        print(doc.page_content)
        
    context = "\n\n".join(doc.page_content for doc in docs)
    
    
    message = prompt.invoke({
        "context":context,
        "question":question
    })
    
    
    # message = state["message"]
    response = llm.invoke(message)
    
    return {"messages": [response]}

# checkpointer

checkpointer = InMemorySaver()

# langgraph

graph = StateGraph(messagedict)


graph.add_node("chat_message", chat_message)
graph.add_edge(START, "chat_message")
graph.add_edge("chat_message", END)


# compile

chatbot = graph.compile(checkpointer)

CONFIG = {"configurable":{"thread_id": "thread_1"}}


question = input("text:")

response = chatbot.invoke({
    "messages": [HumanMessage(content=question)]}, config=CONFIG)

print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(
    response["messages"][-1].content
)