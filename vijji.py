from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import shutil
import os

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud import aiplatform
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

load_dotenv()

app = FastAPI()
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

retriever1 = None
retriever2 = None

loader_mapping = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader
}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1].lower()
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader_cls = loader_mapping.get(file_ext, TextLoader)
    loader = loader_cls(file_path)
    docs = loader.load()

    global chunks, retriever1, retriever2
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(docs)

    embeddings1 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store1 = FAISS.from_documents(chunks, embeddings1)
    retriever1 = vector_store1.as_retriever()

    aiplatform.init(project="gen-lang-client-0149712260", location="us-central1")
    embeddings2 = VertexAIEmbeddings(model_name="gemini-embedding-001")
    vector_store2 = FAISS.from_documents(docs, embeddings2)
    retriever2 = vector_store2.as_retriever()

    return {"message": f"{len(docs)} document(s) loaded from {file.filename}"}

llm1 = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
llm2 = OllamaLLM(model="llama3")

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:""")

rag_chain1 = (
    RunnableMap({
        "context": lambda x: "\n\n".join([doc.page_content for doc in retriever1.invoke(x["question"])]),
        "question": lambda x: x["question"]
    })
    | prompt
    | llm1
    | RunnableLambda(lambda output: {"answer": output})
)

rag_chain2 = (
    RunnableMap({
        "context": lambda x: "\n\n".join([doc.page_content for doc in retriever2.invoke(x["question"])]),
        "question": lambda x: x["question"]
    })
    | prompt
    | llm2
    | RunnableLambda(lambda output: {"answer": output})
)

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

rag_with_history1 = RunnableWithMessageHistory(
    rag_chain1,
    get_session_history,
    input_messages_key="question",
    history_messages_key="context"
)

rag_with_history2 = RunnableWithMessageHistory(
    rag_chain2,
    get_session_history,
    input_messages_key="question",
    history_messages_key="context"
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/gemini")
def ask_question(req: QuestionRequest, request: Request):
    if retriever1 is None:
        return {"error": "Please upload a document first using /upload."}
    session_id = request.client.host
    result = rag_with_history1.invoke(
        {"question": req.question},
        config={"configurable": {"session_id": session_id}}
    )
    return result

@app.post("/ollama")
def ask_question(req: QuestionRequest, request: Request):
    if retriever2 is None:
        return {"error": "Please upload a document first using /upload."}
    session_id = request.client.host
    result = rag_with_history2.invoke(
        {"question": req.question},
        config={"configurable": {"session_id": session_id}}
    )
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
