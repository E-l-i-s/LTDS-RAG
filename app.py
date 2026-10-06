"""Chat with your PDF - a fully local RAG app.

Stack: Streamlit + LangChain + Chroma + Ollama (Llama 3.1).
"""
import os
import tempfile
import uuid

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- Config (override with environment variables if you like) ----
MODEL_NAME = os.getenv("LLM_MODEL", "llama3.1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ---- Cached model helpers (so Streamlit doesn't reload on every rerun) ----
@st.cache_resource
def get_llm():
    return ChatOllama(model=MODEL_NAME, temperature=0.1, base_url=OLLAMA_URL)


@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)


# ---- Indexing: PDF -> chunks -> embeddings -> Chroma ----
def build_vectorstore(uploaded_file):
    if uploaded_file is None:
        st.error("Please upload a PDF first.")
        return None

    # Save to a temp file so PyPDFLoader can read from a real path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        pages = PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    if not any(p.page_content.strip() for p in pages):
        st.error("No text could be extracted (is this a scanned PDF?).")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"pdf_{uuid.uuid4().hex[:8]}",
    )


# ---- RAG chain (LCEL) ----
def format_docs(docs):
    if not docs:
        return "No relevant context found in the PDF."
    return "\n\n".join(d.page_content for d in docs)


def make_rag_chain(vectorstore, k=4, style="Short"):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    style_hint = (
        "Answer in 2-3 concise sentences."
        if style == "Short"
        else "Give a detailed, well-structured answer."
    )
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant answering questions about a PDF.\n"
        "Use ONLY the context below. If the answer is not in the context, "
        "say \"I couldn't find that in the PDF.\"\n"
        f"{style_hint}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain, retriever


# ---- Streamlit UI ----
def init_state():
    st.session_state.setdefault("vectorstore", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pdf_name", None)


def reset_all():
    vs = st.session_state.get("vectorstore")
    if vs is not None:
        try:
            vs.delete_collection()
        except Exception:
            pass
    st.session_state.vectorstore = None
    st.session_state.history = []
    st.session_state.pdf_name = None


def main():
    st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="wide")
    init_state()

    st.title("📄 Chat with your PDF")
    st.caption(f"Fully local RAG · {MODEL_NAME} via Ollama · Chroma · LangChain")

    with st.sidebar:
        st.header("Settings")
        k = st.slider("Chunks to retrieve (k)", 1, 10, 4)
        style = st.radio("Answer style", ["Short", "Detailed"])
        if st.button("Reset everything"):
            reset_all()
            st.rerun()

    left, right = st.columns([1, 2], gap="large")

    # Left: upload + process
    with left:
        st.subheader("1. Upload")
        uploaded = st.file_uploader("Choose a PDF", type="pdf")
        if st.button("Process PDF", type="primary"):
            with st.spinner("Reading, chunking and embedding..."):
                try:
                    vs = build_vectorstore(uploaded)
                except Exception as e:
                    vs = None
                    st.error(
                        f"Indexing failed: {e}\n\n"
                        f"Is Ollama running and have you run "
                        f"`ollama pull {EMBED_MODEL}`?"
                    )
            if vs is not None:
                st.session_state.vectorstore = vs
                st.session_state.pdf_name = uploaded.name
                st.session_state.history = []
                st.success(f"Ready: {uploaded.name}")

        if st.session_state.vectorstore is not None:
            st.markdown("**Try asking:**")
            st.markdown(
                "- What is the main conclusion of this document?\n"
                "- Summarize the key points.\n"
                "- What are the main recommendations?"
            )

    # Right: chat
    with right:
        st.subheader("2. Ask")
        if st.session_state.vectorstore is None:
            st.info("Upload and process a PDF to start chatting.")
        else:
            question = st.text_input("Your question")
            if st.button("Ask") and question.strip():
                chain, retriever = make_rag_chain(
                    st.session_state.vectorstore, k=k, style=style
                )
                with st.spinner("Thinking..."):
                    try:
                        answer = chain.invoke(question)
                        sources = retriever.invoke(question)
                    except Exception as e:
                        st.error(
                            f"Generation failed: {e}\n\n"
                            f"Is Ollama running and have you run "
                            f"`ollama pull {MODEL_NAME}`?"
                        )
                        answer, sources = None, []
                if answer is not None:
                    st.session_state.history.append(
                        {"q": question, "a": answer, "sources": sources}
                    )

            for item in reversed(st.session_state.history):
                with st.chat_message("user"):
                    st.write(item["q"])
                with st.chat_message("assistant"):
                    st.write(item["a"])
                    with st.expander("View sources"):
                        for i, doc in enumerate(item["sources"], 1):
                            page = doc.metadata.get("page", "?")
                            st.markdown(f"**Excerpt {i}** (page {int(page) + 1 if str(page).isdigit() else page})")
                            st.caption(doc.page_content)


if __name__ == "__main__":
    main()
