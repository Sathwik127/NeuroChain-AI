# 🧠 NeuroChain AI  
**A LangChain-Powered Conversational RAG Assistant with Local + Cloud Embeddings & Dual LLMs**

> Upload documents → Embed using local + cloud embeddings → Ask with Gemini or Ollama → Conversational memory → LangSmith tracing → LangServe-ready FastAPI.

---

## 🚀 Overview

**NeuroChain AI** is a powerful Conversational RAG application built using FastAPI, LangChain, and FAISS. It allows users to upload documents in multiple formats, embeds them using both local and cloud embedding models, and enables querying through two LLM paths — Gemini (cloud) and Ollama (local). Each path has its own retriever and supports multi-turn memory, with LangSmith for full observability and LangServe-ready deployment.

---

## 🌐 Key Features

✅ Supports `.pdf`, `.txt`, `.csv`, `.docx` files  
✅ Dual **embedding pipelines**:
- 🔗 Local: HuggingFace Transformers (`all-mpnet-base-v2`)
- ☁️ Cloud: VertexAI Embeddings (`gemini-embedding-001`) ✅ updated to use chunked context

✅ Dual **LLM Paths**:
- ⚡ Google Gemini 1.5 Flash (Cloud)
- 🦙 Ollama with LLaMA3 (Local)

✅ FAISS vector store for fast semantic search  
✅ Memory-aware via `ChatMessageHistory`  
✅ Fully traced with **LangSmith**  
✅ Deployable via **LangServe**

---

## 🧠 Architecture

```
Upload File
    │
    ├─▶ LangChain DocumentLoader
    │
    ├─▶ RecursiveCharacterTextSplitter
    │
    ├─▶ Embedding 1: HuggingFace → FAISS → Retriever 1 → Gemini
    └─▶ Embedding 2: VertexAI     → FAISS → Retriever 2 → Ollama ✅ fixed to use chunks

Both chains use:
    → ChatPromptTemplate
    → ChatMessageHistory
    → RunnableWithMessageHistory
    → LangSmith Tracing
```

---

## 🧰 Tech Stack

| Component         | Tool                                      |
|------------------|-------------------------------------------|
| API Framework     | FastAPI                                   |
| Document Loaders  | LangChain Community Loaders               |
| Text Splitter     | RecursiveCharacterTextSplitter            |
| Embeddings        | HuggingFace + VertexAI Embeddings         |
| Vector DB         | FAISS                                     |
| LLMs              | Gemini 1.5 Flash, Ollama (LLaMA3)         |
| Memory            | ChatMessageHistory                        |
| Observability     | LangSmith                                 |
| Deployment        | LangServe-ready                           |

---

## 📂 Supported File Types

| Extension | Loader                           |
|-----------|----------------------------------|
| `.pdf`    | `PyPDFLoader`                    |
| `.txt`    | `TextLoader`                     |
| `.csv`    | `CSVLoader`                      |
| `.docx`   | `UnstructuredWordDocumentLoader` |

---

## 📦 API Endpoints

### `/upload`
```http
POST /upload
```
- Accepts uploaded document
- Loads, splits into chunks
- Embeds using both HuggingFace and VertexAI
- Builds 2 FAISS vector stores and retrievers:
  - Retriever1 → Gemini path
  - Retriever2 → Ollama path ✅ now uses chunked embeddings

---

### `/gemini`
```http
POST /gemini
```
- LLM: Google Gemini 1.5 Flash
- Retriever: HuggingFace embeddings
- Memory: Conversation-aware
- Traced with LangSmith

---

### `/ollama`
```http
POST /ollama
```
- LLM: Ollama (LLaMA3)
- Retriever: VertexAI embeddings ✅ now chunked
- Memory: Multi-turn support
- Traced with LangSmith

---

## 🧪 Sample Usage

```bash
# Upload
curl -X POST "http://localhost:8000/upload" -F "file=@sample.pdf"

# Ask Gemini
curl -X POST "http://localhost:8000/gemini" -H "Content-Type: application/json" -d '{"question": "Summarize this document."}'

# Ask Ollama
curl -X POST "http://localhost:8000/ollama" -H "Content-Type: application/json" -d '{"question": "What are the key points?"}'
```

---

## 🧠 Memory Handling

Each request is bound to a session (`client.host`) and maintains memory using `ChatMessageHistory`. This allows:
- Context retention across multiple questions
- Smarter, history-aware answers
- Chain input/output linked to previous turns

---

## 📈 LangSmith Observability

LangSmith captures:
- The entire RAG chain flow
- Input → retrieval → LLM response
- Chain timings, tokens, memory state
- Helps debug irrelevant context or prompt issues

---

## 🌍 LangServe Deployment

To deploy a chain:
```python
from langserve import add_routes
add_routes(app, rag_with_history1, path="/api/gemini-chat")
```



---

## 🔐 Environment Setup

Create a `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key
```

---

## ⚙️ Setup Instructions

```bash
pip install -r requirements.txt

# If using Ollama
ollama pull llama3

# Start the server
uvicorn main:app --reload
```

---

## 📁 Project Structure

```
├── main.py                  # FastAPI + LangChain RAG logic
├── uploaded_docs/           # Stores uploaded files
├── .env                     # API keys
├── requirements.txt         # Dependencies
├── README.md                # You are here
```

##Screenshots
The langserve interface looks like this
<img width="1899" height="1012" alt="image" src="https://github.com/user-attachments/assets/016b4500-11c3-46d8-9f6b-1d6491a9a596" />
you can upload the document and you can do execute here
<img width="1919" height="912" alt="image" src="https://github.com/user-attachments/assets/f9428e24-d690-4175-990f-05da4f515cb9" />
you can ask any question about the document 
<img width="1917" height="960" alt="image" src="https://github.com/user-attachments/assets/2dd8c96d-fa2f-464e-a34f-767dc904dd6f" />
<img width="1900" height="934" alt="image" src="https://github.com/user-attachments/assets/04342cea-5681-4228-9057-099ad9d54b82" />
you can do the tracing thorugh langsmith
<img width="1919" height="1052" alt="image" src="https://github.com/user-attachments/assets/1a63e6fc-ce54-4e11-8cc8-115f533bbeef" />


