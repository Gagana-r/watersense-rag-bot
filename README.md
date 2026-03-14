# 💧 WaterSense QA Bot — RAG-based Water Quality Assistant

An AI-powered Q&A chatbot that answers questions about water quality using Retrieval-Augmented Generation (RAG). Built with LangChain, FAISS, and Google Gemini API.

## 🔗 Links
- **GitHub:** https://github.com/Gagana-r/watersense-rag-bot

## 📌 About
WaterSense QA Bot is a domain-specific AI assistant that answers natural language questions about water quality parameters, WHO guidelines, treatment methods, and research findings. It grounds all answers in your own documents — reducing hallucination compared to vanilla LLMs.

## 🛠️ Tech Stack
| Layer | Tool |
|---|---|
| LLM | Google Gemini API (gemini-2.0-flash) |
| RAG Framework | LangChain |
| Vector Database | FAISS (local) |
| Embeddings | HuggingFace sentence-transformers (all-MiniLM-L6-v2) |
| Frontend | Streamlit |

## ✨ Features
- RAG pipeline ingesting domain documents (PDF/TXT)
- Semantic search using FAISS vector store
- Source citation — shows which document the answer came from
- Conversational chat interface with history
- Beautiful UI with water-themed design
- Powered by Google Gemini 2.0 Flash

## 💬 Example Questions
- What is the safe pH range for drinking water?
- What are WHO guidelines for nitrates?
- How is turbidity measured?
- What are the health effects of heavy metals in water?
- What are common water treatment methods?
- What is the acceptable TDS for drinking water?

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Gagana-r/watersense-rag-bot.git
cd watersense-rag-bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create a `.env` file in the root folder:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
Get your free API key at: https://aistudio.google.com/apikey

### 4. Add your documents
Place your water quality PDF or TXT files in the `docs/` folder.

### 5. Run the app
```bash
streamlit run watersense.py
```

### 6. Open in browser
```
http://localhost:8501
```

## 📁 Project Structure
```
watersense-rag-bot/
├── watersense.py          # Main Streamlit app
├── requirements.txt       # Python dependencies
├── .env                   # API key (not uploaded to GitHub)
├── .gitignore             # Ignores .env and cache files
└── docs/                  # Knowledge base documents
    └── water_quality_guidelines.txt
```

## 🧠 How RAG Works
1. **Ingestion** — Documents are loaded and split into chunks
2. **Embedding** — Chunks are converted to vectors using sentence-transformers
3. **Storage** — Vectors are stored in FAISS index
4. **Retrieval** — User question is matched to relevant chunks
5. **Generation** — Gemini API generates answer grounded in retrieved chunks
6. **Citation** — Source documents are displayed with the answer

## 👩‍💻 Author
**Gagana R** — [GitHub](https://github.com/Gagana-r) | [LinkedIn](https://www.linkedin.com/in/gagana-sri-r-518488265/)
