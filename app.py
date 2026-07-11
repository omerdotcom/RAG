import streamlit as st
from pathlib import Path
import os

from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

DATA_DIR = "data"
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "q0.6"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 6
# ===================================================

@st.cache_resource
def load_vectorstore():
    """Load or create the vectorstore"""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    if os.path.exists(CHROMA_PATH) and len(list(Path(CHROMA_PATH).glob("**/*"))) > 0:
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        st.success(f"✅ Loaded existing vectorstore with {vectorstore._collection.count()} chunks")
    else:
        st.info("No vectorstore found. Please ingest documents first.")
        vectorstore = None
    return vectorstore


def ingest_documents():
    """Ingest all documents from data/ folder"""
    with st.spinner("Loading and processing documents..."):
        # Load documents
        docs = []
        if Path(DATA_DIR).exists():
            pdf_loader = PyPDFDirectoryLoader(DATA_DIR)
            docs.extend(pdf_loader.load())
        
        for ext in ["*.txt", "*.md"]:
            for file in Path(DATA_DIR).glob(ext):
                loader = TextLoader(str(file), encoding="utf-8")
                docs.extend(loader.load())
        
        if not docs:
            st.error("No documents found in the 'data/' folder!")
            return None
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        
        # Create vectorstore
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        
        st.success(f"✅ Successfully ingested {len(chunks)} chunks from {len(docs)} documents!")
        return vectorstore


def create_rag_chain(vectorstore):
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.3,
        num_ctx=8192,
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    
    template = """You are a helpful, accurate assistant. Use ONLY the following context to answer the question.
If the context doesn't contain enough information, clearly say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# ====================== STREAMLIT UI ======================
st.title("📚 Local Private RAG Assistant")
st.caption("Fully offline • Powered by Ollama • Your documents, your privacy")

# Sidebar
with st.sidebar:
    st.header("Controls")
    
    if st.button("🔄 Ingest Documents from data/ folder", type="primary"):
        vectorstore = ingest_documents()
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.rerun()
    
    st.divider()
    
    st.markdown("### Settings")
    selected_model = st.selectbox(
        "LLM Model",
        ["llama3.1:8b", "qwen2.5:14b", "mistral-nemo:12b"],
        index=0
    )
    top_k = st.slider("Number of retrieved chunks (k)", 3, 12, TOP_K)
    
    st.divider()
    st.markdown("**Instructions:**\n"
                "1. Put your PDFs, .txt, or .md files in the `data/` folder\n"
                "2. Click 'Ingest Documents'\n"
                "3. Start asking questions below")

# Main area
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()

if st.session_state.vectorstore is None:
    st.warning("Please ingest your documents first using the sidebar button.")
else:
    # Create chain once vectorstore is available
    if "rag_chain" not in st.session_state:
        with st.spinner("Building RAG chain..."):
            st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore)
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask anything about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(prompt)
                st.markdown(response)
        
        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.divider()
st.caption("💡 Pro tip: This entire app runs locally with Ollama. No data leaves your computer.")
