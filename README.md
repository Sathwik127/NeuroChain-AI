# 🔮 Omni-Chat AI: A Multi-Source, Conversational RAG Assistant with Local & Cloud LLMs

Welcome to **Omni-Chat AI** – a powerful, modular Conversational RAG (Retrieval-Augmented Generation) system that combines the strength of **local** and **cloud-based** LLMs for answering queries based on diverse knowledge sources. Built using **LangChain**, **FAISS**, **FastAPI**, and integrated with **LangSmith**, this assistant is designed for high flexibility and real-world deployment.

---

## 🚀 Project Goals

- ✅ Ingest and vectorize documents of various formats.
- ✅ Enable **conversational retrieval** using memory-aware RAG chains.
- ✅ Dynamically route to **local (Ollama)** or **cloud (Google Gemini)** LLMs.
- ✅ Deploy an interactive API using FastAPI.
- ✅ Monitor and debug interactions with **LangSmith**.

---

## 📁 Project Structure

📦 omni-chat-ai/
├── app.py # FastAPI app with document ingestion & chat endpoints
├── .env # API keys and config (not committed)
├── requirements.txt # Python dependencies
├── uploaded_docs/ # Uploaded source documents
├── README.md # You're here!

markdown
Copy
Edit

---

## 📚 Key Features

### 📥 1. Document Ingestion & Chunking

- Supports: **PDF**, **TXT**, **CSV**, **DOCX**
- Uses `RecursiveCharacterTextSplitter` to chunk documents.
- Embeds using:
  - 🔹 `HuggingFace (all-mpnet-base-v2)`
  - 🔸 `Google VertexAI (gemini-embedding-001)`
- Stores in FAISS vector store.
- Automatically creates **two retrievers** for hybrid LLM use.

### 💬 2. Conversational RAG Chains

- Two fully functional RAG pipelines:
  - 🧠 **Gemini** (Cloud) + `retriever1`
  - 🐘 **Ollama (LLaMA3)** (Local) + `retriever2`
- Uses `RunnableWithMessageHistory` for context-aware conversations.
- Prompt templates customized for clarity and consistency.

### ⚙️ 3. API Endpoints (FastAPI)

| Endpoint         | Description                         |
|------------------|-------------------------------------|
| `/upload`        | Upload and process documents        |
| `/gemini`        | Ask a question using Gemini + retriever1 |
| `/ollama`        | Ask a question using LLaMA3 + retriever2 |

### 🧪 4. Observability with LangSmith

- Tracks all runs, chains, and interactions.
- Helps debug memory, retrieval, and prompt performance.

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/omni-chat-ai.git
cd omni-chat-ai
2. Create & Activate Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
4. Setup .env
Create a .env file in the root directory:

env
Copy
Edit
GEMINI_API_KEY=your_google_generativeai_api_key
⚠️ Note: Ensure you’ve enabled Vertex AI and Gemini access in your Google Cloud Project.

5. Run the App
bash
Copy
Edit
uvicorn app:app --reload --host localhost --port 8000
📡 API Usage Examples
Upload a File
bash
Copy
Edit
curl -X POST "http://localhost:8000/upload" -F "file=@sample.pdf"
Ask a Question (Gemini)
bash
Copy
Edit
curl -X POST "http://localhost:8000/gemini" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the summary of the document?"}'
Ask a Question (Ollama)
bash
Copy
Edit
curl -X POST "http://localhost:8000/ollama" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the main topics?"}'
📦 requirements.txt
Here’s the list of required Python packages (create requirements.txt):

nginx
Copy
Edit
fastapi
uvicorn
langchain
langchain-community
langchain-core
langchain-google-genai
langchain-google-vertexai
langchain-ollama
huggingface-hub
sentence-transformers
python-dotenv
faiss-cpu
google-cloud-aiplatform
pydantic
shutil
Run this to install all:

bash
Copy
Edit
pip install -r requirements.txt
🧠 LLMs Used
🌐 Google Gemini 1.5 Flash – via langchain_google_genai

🖥️ Ollama LLaMA3 – via langchain_ollama

📊 Observability: LangSmith
LangSmith is enabled to:

Trace multi-turn conversation flows.

Identify prompt failures or bad retrieval.

Debug how memory, retriever, and model interact.

To view traces, visit your LangSmith dashboard (requires LangChain+LangSmith setup).

🧪 Future Enhancements (Suggestions)
🔄 Add MultiQueryRetriever for better query coverage.

🧵 Enable LangServe for scalable deployment.

🧠 Fine-tune prompts with few-shot learning.

📁 Add support for more document formats (e.g., HTML, JSON, Markdown).

🌐 Dockerize for container-based deployment.

📄 License
This project is intended for educational and academic purposes.
All third-party tools and models used adhere to their respective licenses.

🙌 Acknowledgments
Special thanks to:

LangChain

Hugging Face

Google VertexAI & Gemini

Ollama

LangSmith



1. **Create a file named `README.md`** in your project root (same level as `app.py`).
2. **Paste** the full content above into that file.
3. **Update the placeholders**:
   - `[Your Name]`
   - `[Your Institution or Course]`
   - `@your-username` (GitHub handle)
   - `your_google_generativeai_api_key`
4. **Commit and push to GitHub**:

```bash
git add README.md
git commit -m "Add detailed README with project documentation"
git push origin main
