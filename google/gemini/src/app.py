import os
import google.generativeai as genai
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Importações do LangChain e FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from env import MY_KEY
# --- Configuração do Gemini e Variáveis de Ambiente ---
load_dotenv()
if MY_KEY:
    os.environ['GOOGLE_API_KEY'] = MY_KEY
    print("Chave de API carregada com sucesso.")
else:
    print("A chave de API não foi encontrada no arquivo .env.")

genai.configure(api_key=MY_KEY)

# --- Instância do FastAPI ---
app = FastAPI(title="PDF Chat com RAG e FastAPI")

# Dicionário para armazenar o vector_store FAISS em memória por sessão
vector_store_sessions = {}

# --- Modelos Pydantic ---
class Question(BaseModel):
    doc_id: str
    question: str

# --- Funções Auxiliares ---
def get_pdf_text(file: UploadFile) -> str:
    """Extrai texto de um arquivo PDF usando PyMuPDF."""
    try:
        doc = fitz.open(stream=file.file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar o PDF: {e}")

def get_text_chunks(text: str) -> list:
    """Divide o texto em chunks menores para embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks: list):
    """Cria embeddings e um vetor FAISS a partir dos chunks de texto."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# --- Endpoints da API ---

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint para fazer o upload de um PDF e processá-lo para ser usado na conversa.
    O PDF é dividido em chunks e um vetor FAISS é criado.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são permitidos.")

    raw_text = get_pdf_text(file)
    chunks = get_text_chunks(raw_text)
    vector_store = get_vector_store(chunks)

    # Gera um ID de sessão e armazena o vector_store em memória
    doc_id = "session_" + os.urandom(8).hex()
    vector_store_sessions[doc_id] = vector_store

    return {"message": "PDF processado e vector store criado com sucesso!", "doc_id": doc_id}

@app.post("/ask/")
async def ask_question(question_data: Question):
    """
    Endpoint para fazer uma pergunta sobre o conteúdo do PDF usando a técnica de RAG.
    Ele busca os chunks relevantes do vetor e os usa para gerar a resposta.
    """
    doc_id = question_data.doc_id
    question = question_data.question

    if doc_id not in vector_store_sessions:
        raise HTTPException(status_code=404, detail="ID de sessão não encontrado. Faça o upload do PDF novamente.")

    vector_store = vector_store_sessions[doc_id]

    # Busca os chunks mais relevantes para a pergunta
    relevant_docs = vector_store.similarity_search(question)

    # Configura o modelo e a cadeia de QA (Question-Answering)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    # Prompt de instrução para a IA
    prompt_template = """
    Responda a pergunta EXCLUSIVAMENTE com base no contexto fornecido.
    Se a resposta não estiver no contexto, diga "A informação não está disponível neste documento."
    
    Contexto:
    {context}
    
    Pergunta: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=PROMPT)
    
    # Executa a cadeia com os documentos relevantes
    response = chain.invoke({"input_documents": relevant_docs, "question": question})

    return {"answer": response["output_text"]}

