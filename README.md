# 🎥 YouTube RAG Chatbot

Ask questions about any YouTube video using AI. Paste a URL, get answers.

**Live Demo:** https://mzn3vcm7skmgydjrgqgozd.streamlit.app/

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Llama 3.1 8B (Groq) |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Vector Store | FAISS |
| Transcripts | Supadata API |
| Orchestration | LangChain |

---

## Architecture

```
YouTube URL → Supadata API → Transcript → Text Splitting
                                                ↓
                                        HuggingFace Embeddings
                                                ↓
                                        FAISS Vector Store
                                                ↓
User Question → Semantic Search → Relevant Chunks → Groq LLM → Answer
```

---

## Installation

1. Clone the repo
```bash
git clone https://github.com/harshjoshi297/RAG.git
cd RAG
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file
```
HUGGINGFACEHUB_API_TOKEN=your_token
SUPADATA_API_KEY=your_key
GROQ_API_KEY=your_key
```

4. Run the app
```bash
streamlit run main2.py
```
