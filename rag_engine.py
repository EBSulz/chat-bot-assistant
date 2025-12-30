"""
RAG (Retrieval-Augmented Generation) engine for question answering.
"""

import os
from typing import List, Tuple

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from document_processor import DocumentProcessor


class RAGEngine:
    """RAG engine for document-based question answering."""
    
    def __init__(self, documents_dir: str = "documents", top_k: int = 3):
        """
        Initialize the RAG engine.
        
        Args:
            documents_dir: Path to the documents directory
            top_k: Number of document chunks to retrieve for each query
        """
        self.documents_dir = documents_dir
        self.top_k = top_k
        self.embeddings = None
        self.vector_store = None
        self.chunks = None
        self.llm = None
        self._initialize()
    
    def _initialize(self):
        """Initialize embeddings, LLM, and process documents."""
        # Get API key from environment or streamlit secrets
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("OPENAI_API_KEY")
            except:
                pass
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in .streamlit/secrets.toml "
                "or as an environment variable."
            )
        
        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,  # Low temperature for more deterministic responses
            openai_api_key=api_key
        )
        
        # Process documents and create vector store
        self._build_vector_store()
    
    def _build_vector_store(self):
        """Load documents, create embeddings, and build FAISS vector store."""
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        self.chunks = processor.process_documents(self.documents_dir)
        
        if not self.chunks:
            self.vector_store = None
            return
        
        # Create embeddings for all chunks
        texts = [chunk.page_content for chunk in self.chunks]
        embedding_dimension = len(self.embeddings.embed_query("test"))
        
        # Get embeddings for all chunks
        chunk_embeddings = self.embeddings.embed_documents(texts)
        
        # Convert to numpy array
        embeddings_array = np.array(chunk_embeddings).astype("float32")
        
        # Create FAISS index
        index = faiss.IndexFlatL2(embedding_dimension)
        index.add(embeddings_array)
        
        self.vector_store = index
    
    def _retrieve_relevant_chunks(self, query: str) -> List[Document]:
        """
        Retrieve relevant document chunks for a given query.
        
        Args:
            query: User query string
            
        Returns:
            List of relevant Document chunks
        """
        if not self.vector_store or not self.chunks:
            return []
        
        # Embed the query
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype("float32")
        
        # Search for similar chunks
        distances, indices = self.vector_store.search(query_vector, k=self.top_k)
        
        # Retrieve the corresponding document chunks
        relevant_chunks = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        
        return relevant_chunks
    
    def query(self, question: str) -> Tuple[str, List[Document]]:
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (answer, source_chunks)
        """
        if not self.vector_store or not self.chunks:
            return (
                "No documents have been loaded. Please add a document to the documents/ directory.",
                []
            )
        
        # Retrieve relevant chunks
        relevant_chunks = self._retrieve_relevant_chunks(question)
        
        if not relevant_chunks:
            return (
                "I couldn't find relevant information in the document to answer your question.",
                []
            )
        
        # Build context from retrieved chunks
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        
        # Create prompt with strict anti-hallucination instructions
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based ONLY on the provided context.

IMPORTANT RULES:
1. Answer ONLY using information from the provided context below.
2. If the context does not contain enough information to answer the question, explicitly say: "I don't have information about that in my document."
3. Do NOT make up, infer, or generate any information that is not explicitly stated in the context.
4. If asked about something not in the context, politely state that you don't have that information.
5. Be concise and accurate in your responses.

Context:
{context}"""),
            ("human", "{question}")
        ])
        
        # Create chain
        chain = prompt_template | self.llm | StrOutputParser()
        
        # Generate response
        try:
            answer = chain.invoke({
                "context": context,
                "question": question
            })
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
        
        return answer, relevant_chunks
    
    def has_documents(self) -> bool:
        """Check if documents have been loaded."""
        return self.vector_store is not None and len(self.chunks) > 0

