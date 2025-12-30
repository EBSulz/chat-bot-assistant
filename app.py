"""
Streamlit app for RAG-based chatbot.
"""

import os
import streamlit as st
from pathlib import Path

from rag_engine import RAGEngine
from document_processor import DocumentProcessor


# Page configuration
st.set_page_config(
    page_title="Get to know Eduardo",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False


def initialize_rag_engine():
    """Initialize or reinitialize the RAG engine."""
    try:
        # Check if documents exist
        documents_dir = Path("documents")
        if documents_dir.exists():
            files = list(documents_dir.glob("*.*"))
            # Filter out .gitkeep
            files = [f for f in files if f.name != ".gitkeep"]
            if not files:
                return None, "No documents found. Please add a document to get started."
        
        # Initialize RAG engine
        rag_engine = RAGEngine(documents_dir="documents", top_k=3)
        
        if not rag_engine.has_documents():
            return None, "No documents could be processed. Please check your document format."
        
        return rag_engine, None
    except Exception as e:
        return None, f"Error initializing RAG engine: {str(e)}"


def process_uploaded_file(uploaded_file):
    """Save uploaded file to documents directory."""
    documents_dir = Path("documents")
    documents_dir.mkdir(exist_ok=True)
    
    file_path = documents_dir / uploaded_file.name
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


# Sidebar for document management
with st.sidebar:
    st.title("📄 Document Management")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "md", "pdf"],
        help="Upload your biographical document (life, academic track, work track)"
    )
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    file_path = process_uploaded_file(uploaded_file)
                    st.success(f"Document '{uploaded_file.name}' uploaded successfully!")
                    # Reset RAG engine to trigger rebuild
                    st.session_state.rag_engine = None
                    st.session_state.documents_processed = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error uploading file: {str(e)}")
    
    st.markdown("---")
    
    # Document list
    st.subheader("Current Documents")
    documents_dir = Path("documents")
    if documents_dir.exists():
        files = list(documents_dir.glob("*.*"))
        files = [f for f in files if f.name != ".gitkeep"]
        if files:
            for file in files:
                st.text(f"📄 {file.name}")
        else:
            st.info("No documents found. Upload a document to get started.")
    else:
        st.info("Documents directory not found.")
    
    st.markdown("---")
    
    # Initialize/Reinitialize button
    if st.button("🔄 Reinitialize Chatbot", help="Rebuild the vector store from documents"):
        with st.spinner("Initializing chatbot..."):
            st.session_state.rag_engine = None
            st.session_state.documents_processed = False
            rag_engine, error = initialize_rag_engine()
            if rag_engine:
                st.session_state.rag_engine = rag_engine
                st.session_state.documents_processed = True
                st.success("Chatbot initialized successfully!")
            else:
                st.error(error or "Failed to initialize chatbot.")
            st.rerun()


# Main chat interface
st.title("💬 Get to know Eduardo")
st.markdown("Ask me anything about the information in my document!")

# Initialize RAG engine if not already done
if st.session_state.rag_engine is None and not st.session_state.documents_processed:
    with st.spinner("Initializing chatbot..."):
        rag_engine, error = initialize_rag_engine()
        if rag_engine:
            st.session_state.rag_engine = rag_engine
            st.session_state.documents_processed = True
        elif error:
            st.error(error)
            st.info("Please add a document to the 'documents' directory or upload one using the sidebar.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Source Chunks"):
                for i, chunk in enumerate(message["sources"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content)
                    st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    # Check if RAG engine is initialized
    if st.session_state.rag_engine is None:
        st.error("Chatbot is not initialized. Please add a document first.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, source_chunks = st.session_state.rag_engine.query(prompt)
                st.markdown(answer)
                
                # Display source chunks if available
                if source_chunks:
                    with st.expander("📚 Source Chunks Used"):
                        for i, chunk in enumerate(source_chunks, 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content)
                            if hasattr(chunk, "metadata") and chunk.metadata.get("source"):
                                st.caption(f"Source: {chunk.metadata['source']}")
                            st.markdown("---")
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                st.error(error_message)
                answer = error_message
                source_chunks = []
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": source_chunks
    })

# Clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

