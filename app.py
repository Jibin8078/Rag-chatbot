# app.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
import jinja2


# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------------------
#  CREATE FASTAPI APP
# ---------------------------
app = FastAPI(title="RAG + LLaMA API")

# ---------------------------
#  FRONTEND CONFIG
# ---------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------------------
# LOAD DOCUMENTS
# ---------------------------
loader = TextLoader("data.txt")
documents = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = splitter.split_documents(documents)

# ---------------------------
#  VECTOR STORE
# ---------------------------
embeddings = OllamaEmbeddings(model="all-minilm")
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}  # instead of default 4–5
)

# ---------------------------
#  LLM
# ---------------------------
llm = OllamaLLM(
    model="phi3",
    temperature=0
)

# ---------------------------
# 6️⃣ PROMPT
# ---------------------------
prompt = PromptTemplate.from_template(
    """You are a QA assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say:
"I don't know based on the document."

Context:
{context}

Question:
{question}

Answer:
"""
)

# ---------------------------
# 7️⃣ RAG CHAIN
# ---------------------------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ---------------------------
# 8️⃣ API MODEL
# ---------------------------
class Question(BaseModel):
    query: str

# ---------------------------
# 9️⃣ API ENDPOINT
# ---------------------------
@app.post("/ask")
def ask(q: Question):
    answer = rag_chain.invoke(q.query)
    return {"answer": answer}

#prevent request delay

@app.on_event("startup")
def warmup():
    llm.invoke("Hello")
# ---------------------------
# 🔟 FRONT WEB PAGE
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_answer(question: str):
    return rag_chain.invoke(question)

@app.post("/ask")
async def ask(q: Question):
    return {"answer": cached_answer(q.query)}