RAG + LLaMA API (FastAPI + FAISS + Ollama)

A Retrieval-Augmented Generation (RAG) application built using:

1 FastAPI

2 LangChain

3 FAISS

4 Ollama

This system reads a local document (data.txt), converts it into embeddings, retrieves relevant chunks using vector similarity search, and generates answers using a local LLM.

The model answers strictly from context to reduce hallucination.

Features

* Document loading and text chunking

* Embeddings using all-minilm (Ollama)

* FAISS vector search

* Retrieval-Augmented Generation (RAG)

* Strict anti-hallucination prompt

* LRU caching for repeated queries

* FastAPI REST API

* Simple HTML frontend

  

Architecture
  
  
User Question
      ↓
FAISS Retriever (Top K=2)
      ↓
Prompt Template (Strict Context Mode)
      ↓
Phi-3 LLM (temperature=0)
      ↓
Final Answer


Project Structure


.
│── app.py
│── data.txt
│── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│
└── README.md



1 Create a Virtual Environment
   python -m venv venv

2 Install Dependencies
   pip install -r requirements.txt

3 Or install manually
   pip install fastapi uvicorn langchain langchain-community langchain-core langchain-text-splitters langchain-ollama faiss-cpu jinja2

4 Install and Setup Ollama
   Download from:
   https://ollama.com

5  Pull required models:
    ollama pull phi3
    ollama pull all-minilm

6 Start the ollma server
     ollama serve

7 Run the Application
  uvicorn app:app --reload

Server will start at:http://127.0.0.1:8000

