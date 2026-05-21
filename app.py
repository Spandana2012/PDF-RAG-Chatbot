# -------------------------------
# IMPORTS
# -------------------------------

import os
import tempfile
import json
import hashlib
from pathlib import Path

from dotenv import load_dotenv

import streamlit as st

from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import errors as genai_errors

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------------
# LOAD ENV
# -------------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🌌",
    layout="wide"
)


# -------------------------------
# COSMIC UI
# -------------------------------

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at top left, #1e1b4b, transparent 30%),
        radial-gradient(circle at bottom right, #312e81, transparent 30%),
        linear-gradient(to bottom, #020617, #050816);
    color: white;
}

.chat-title {
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(90deg, #c084fc, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.chat-subtitle {
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.source-badge {
    display: inline-block;
    padding: 6px 12px;
    margin: 5px;
    border-radius: 12px;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(168,85,247,0.4);
    color: white;
    font-size: 14px;
}

[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 12px;
    margin: 10px 0;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(15,23,42,0.65);
    backdrop-filter: blur(12px);
}

[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.85);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.stButton button {
    border-radius: 12px;
    background: linear-gradient(90deg,#7c3aed,#2563eb);
    color: white;
    border: none;
    font-weight: 600;
}

.stButton button:hover {
    background: linear-gradient(90deg,#8b5cf6,#3b82f6);
}

</style>
""", unsafe_allow_html=True)


# -------------------------------
# TITLE
# -------------------------------

st.markdown(
    '<div class="chat-title">🌌 PDF RAG Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="chat-subtitle">Upload PDFs and chat with them using Gemini + RAG</div>',
    unsafe_allow_html=True
)


# -------------------------------
# CHAT HISTORY FOLDER
# -------------------------------

HISTORY_DIR = Path("chat_history")
HISTORY_DIR.mkdir(exist_ok=True)


# -------------------------------
# HELPERS
# -------------------------------

def get_history_file(history_key):

    safe_name = "".join(
        c for c in history_key
        if c.isalnum() or c in (" ", "-", "_")
    ).strip()

    return HISTORY_DIR / f"{safe_name}.json"


def save_chat_history(history_key, messages):

    history_file = get_history_file(history_key)

    with open(history_file, "w") as f:
        json.dump(messages, f, indent=2)


def get_uploaded_pdf_key(uploaded_files):

    pdf_signatures = []

    for pdf_file in uploaded_files:

        file_digest = hashlib.sha256(
            pdf_file.getvalue()
        ).hexdigest()

        pdf_signatures.append(
            f"{pdf_file.name}:{pdf_file.size}:{file_digest}"
        )

    upload_signature = "|".join(
        sorted(pdf_signatures)
    )

    return hashlib.sha256(
        upload_signature.encode("utf-8")
    ).hexdigest()


# -------------------------------
# SIDEBAR
# -------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    if api_key:
        st.success("✅ API Key loaded")
    else:
        st.error("❌ GOOGLE_API_KEY missing")

    model_name = st.selectbox(
        "Choose Gemini model",
        [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash"
        ],
        index=0
    )

    st.divider()

    clear_chat = st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    )


# -------------------------------
# SESSION STATE
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "current_pdf_key" not in st.session_state:
    st.session_state.current_pdf_key = None


# -------------------------------
# CLEAR CHAT
# -------------------------------

if clear_chat:

    st.session_state.messages = []

    st.rerun()


# -------------------------------
# EMBEDDINGS
# -------------------------------

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -------------------------------
# BUILD VECTORSTORE
# -------------------------------

def build_vectorstore(uploaded_files):

    all_docs = []

    for pdf_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_pdf:

            temp_pdf.write(
                pdf_file.getvalue()
            )

            temp_pdf_path = temp_pdf.name

        try:

            loader = PyPDFLoader(
                temp_pdf_path
            )

            docs = loader.load()

        finally:

            os.remove(temp_pdf_path)

        for doc in docs:

            doc.metadata["source_file"] = pdf_file.name

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)

    vectorstore = Chroma.from_documents(
        chunks,
        get_embeddings()
    )

    return vectorstore, chunks


# -------------------------------
# FILE UPLOAD
# -------------------------------

uploaded_files = st.file_uploader(
    "📄 Upload PDF(s)",
    type="pdf",
    accept_multiple_files=True
)


# -------------------------------
# BUILD VECTORSTORE
# -------------------------------

if uploaded_files:

    uploaded_pdf_key = get_uploaded_pdf_key(
        uploaded_files
    )

    pdfs_changed = (
        st.session_state.current_pdf_key
        != uploaded_pdf_key
    )

    if pdfs_changed:

        st.session_state.vectorstore = None
        st.session_state.messages = []

        st.session_state.current_pdf_key = uploaded_pdf_key

    if st.session_state.vectorstore is None:

        with st.spinner(
            "📚 Reading PDFs and creating embeddings..."
        ):

            vectorstore, chunks = build_vectorstore(
                uploaded_files
            )

            st.session_state.vectorstore = vectorstore

        st.success(
            f"✅ Indexed successfully ({len(chunks)} chunks)"
        )


# -------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📌 Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f'<div class="source-badge">{source}</div>',
                        unsafe_allow_html=True
                    )


# -------------------------------
# CHAT INPUT
# -------------------------------

question = st.chat_input(
    "Ask about your PDF..."
)


# -------------------------------
# QUESTION ANSWERING
# -------------------------------

if question and uploaded_files:

    # USER MESSAGE

    user_message = {
        "role": "user",
        "content": question
    }

    st.session_state.messages.append(
        user_message
    )

    with st.chat_message("user"):

        st.markdown(question)

    # ASSISTANT RESPONSE

    with st.chat_message("assistant"):

        with st.spinner("🧠 Thinking..."):

            try:

                retriever = (
                    st.session_state.vectorstore
                    .as_retriever(
                        search_kwargs={"k": 4}
                    )
                )

                try:

                    relevant_docs = (
                        retriever.get_relevant_documents(
                            question
                        )
                    )

                except AttributeError:

                    relevant_docs = retriever.invoke(
                        question
                    )

                # FILTER ONLY CURRENT PDFs

                current_pdf_names = {
                    pdf.name
                    for pdf in uploaded_files
                }

                filtered_docs = []

                for doc in relevant_docs:

                    source_file = doc.metadata.get(
                        "source_file",
                        ""
                    )

                    if source_file in current_pdf_names:

                        filtered_docs.append(doc)

                relevant_docs = filtered_docs

                context = "\n\n".join(
                    doc.page_content
                    for doc in relevant_docs
                )

                if not context.strip():

                    answer = (
                        "I could not find relevant "
                        "information in the uploaded PDF."
                    )

                    sources = []

                else:

                    prompt = f"""
Answer the question using ONLY the context below.

If the answer is not present,
say the PDF does not contain enough information.

Context:
{context}

Question:
{question}
"""

                    client = genai.Client(
                        api_key=api_key
                    )

                    response = (
                        client.models.generate_content(
                            model=model_name,
                            contents=prompt
                        )
                    )

                    answer = response.text.strip()

                    sources = []

                    for doc in relevant_docs:

                        metadata = doc.metadata

                        page_num = (
                            metadata.get("page", 0) + 1
                        )

                        source_file = metadata.get(
                            "source_file",
                            "PDF"
                        )

                        source_text = (
                            f"{source_file} - Page {page_num}"
                        )

                        if source_text not in sources:

                            sources.append(source_text)

            except google_exceptions.PermissionDenied:

                answer = (
                    "🚫 Permission denied for Gemini API."
                )

                sources = []

            except genai_errors.ClientError as error:

                answer = (
                    f"🚫 Gemini client error: {error}"
                )

                sources = []

            except Exception as error:

                answer = (
                    f"🚫 Unexpected error: {error}"
                )

                sources = []

            # DISPLAY RESPONSE

            st.markdown(f"🧠 {answer}")

            if sources:

                with st.expander("📌 Sources"):

                    for source in sources:

                        st.markdown(
                            f'<div class="source-badge">{source}</div>',
                            unsafe_allow_html=True
                        )

    # SAVE MESSAGE

    assistant_message = {
        "role": "assistant",
        "content": f"🧠 {answer}",
        "sources": sources
    }

    st.session_state.messages.append(
        assistant_message
    )