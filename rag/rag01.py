from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings  import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    search_type = "similarity",
    search_kwargs = {"k":3}
)

query = "what is lstm"

result = retriever.invoke(query)

for i ,  doc in enumerate(result):
    print(f"----result---{i+1}")
    print(doc.page_content)