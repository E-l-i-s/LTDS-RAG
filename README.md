# Chat with your PDF (local RAG)

Upload a PDF and ask questions about it. Everything runs locally:
**Streamlit + LangChain + Chroma + Ollama (Llama 3.1)**.

Pipeline: PDF -> chunks (800 chars, 150 overlap) -> embeddings -> Chroma -> retriever -> Llama 3.1 -> answer (+ sources).

## 1. Pull the Ollama models
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```
Make sure Ollama is running (`ollama serve`, or the desktop app).

## 2. Install
```bash
python -m venv .venv
# macOS/Linux:  source .venv/bin/activate
# Windows:      .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run
```bash
streamlit run app.py
```

## Usage
1. Upload a PDF on the left, click **Process PDF**.
2. Type a question on the right, click **Ask**.
3. Open **View sources** to see the exact excerpts used.
4. Sidebar: set chunks retrieved (k), Short/Detailed answers, or reset.

## Optional config (environment variables)
| Variable | Default |
|---|---|
| `LLM_MODEL` | `llama3.1` |
| `EMBED_MODEL` | `nomic-embed-text` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` |

## Troubleshooting
- **Connection error**: Ollama isn't running.
- **model not found**: run the `ollama pull` commands above.
- **No text extracted**: the PDF is likely scanned images (needs OCR).
- Uses an in-memory Chroma store, so indexes reset when you restart the app.
