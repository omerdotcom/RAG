import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ====================== CONFIG ======================
DATA_DIR = "data"                    # put your PDFs, .txt, .md here
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "q0.6"            # change to your preferred model
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 6
# ===================================================

def load_documents():
    """Load all documents from data directory"""
    docs = []
    
    # PDFs
    if Path(DATA_DIR).exists():
        pdf_loader = PyPDFDirectoryLoader(DATA_DIR)
        docs.extend(pdf_loader.load())
    
    # Add support for .txt / .md if needed
    for ext in ["*.txt", "*.md"]:
        for file in Path(DATA_DIR).glob(ext):
            loader = TextLoader(str(file), encoding="utf-8")
            docs.extend(loader.load())
    
    print(f"Loaded {len(docs)} documents")
    return docs


def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_documents(docs)


def create_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Create or load existing DB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        # Optional but recommended for consistency:
        collection_name="local_rag_collection"   # give it a fixed name
    )
    print(f"Created new vectorstore with {len(chunks)} chunks")
    
    return vectorstore


def create_rag_chain(vectorstore):
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.3,
        num_ctx=8192,          # increase if your model supports it
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    
    # Strong system prompt for 2026 local RAG
    template = """You are a helpful, accurate assistant. Answer the question using ONLY the provided context.
If you don't know or the context doesn't contain the answer, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # LCEL RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


# ====================== MAIN ======================
if __name__ == "__main__":
    # 1. Ingest documents (run once or when data changes)
    print("Loading documents...")
    raw_docs = load_documents()
    chunks = split_documents(raw_docs)
    
    print("Creating/updating vectorstore...")
    vectorstore = create_vectorstore(chunks)
    
    # 2. Create RAG chain
    print("Building RAG chain...")
    chain = create_rag_chain(vectorstore)
    
    # 3. Interactive query loop
    print("\n✅ Local RAG is ready! Type your questions (or 'exit' to quit)\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            break
        if not query:
            continue
            
        print("Thinking...")
        answer = chain.invoke(query)
        print(f"\nAssistant: {answer}\n")
