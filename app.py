import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

FAISS_INDEX_PATH = "faiss_index_projeto" 

@st.cache_resource
def get_embeddings_model():
    print("Carregando modelo de embedding (HuggingFace)...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    print("Modelo de embedding carregado.")
    return embeddings

@st.cache_resource
def get_llm():
    print("Carregando LLM (Gemini)...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        google_api_key=google_api_key,
        convert_system_message_to_human=True
    )
    print("LLM carregado.")
    return llm

@st.cache_resource
def load_faiss_index(embeddings_model):
    print("Carregando índice FAISS local...")
    db = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings_model, 
        allow_dangerous_deserialization=True
    )
    print("Índice FAISS carregado.")
    return db

@st.cache_resource
def get_rag_chain(_llm, _retriever):
    print("Criando a cadeia RAG...")
    prompt_template = """
    Você é um assistente especialista. Responda a pergunta *apenas* com base no contexto.
    Contexto: {context}
    Pergunta: {input}
    Resposta:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    document_chain = create_stuff_documents_chain(_llm, prompt)
    retrieval_chain = create_retrieval_chain(_retriever, document_chain)
    print("Cadeia RAG pronta.")
    return retrieval_chain


st.set_page_config(page_title="Chat com Documentos", layout="wide")
st.title("📄 Chatbot com Documentos (Usando Gemini)")

if not google_api_key:
    st.error("GOOGLE_API_KEY não encontrada! Configure-a nos 'Secrets' do Streamlit.")
else:
    try:
        embeddings = get_embeddings_model()
        llm = get_llm()
        
        db = load_faiss_index(embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 4})
        
        chain = get_rag_chain(llm, retriever)

        st.write("O índice está carregado. Faça sua pergunta sobre o documento.")
        
        user_question = st.text_input("Sua pergunta:")

        if user_question:
            with st.spinner("Pensando... (Consultando o Gemini e o índice)"):
                response = chain.invoke({"input": user_question})
                
                st.subheader("Resposta:")
                st.write(response["answer"])

    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar os componentes: {e}")
        st.error("Verifique se a pasta 'faiss_index_projeto' existe no seu repositório.")