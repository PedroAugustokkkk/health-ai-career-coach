# --- Imports ---
import streamlit as st # Biblioteca do Streamlit
import os
from dotenv import load_dotenv

# LLM (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI
# Embeddings (Local/Gratuita)
from langchain_community.embeddings import HuggingFaceEmbeddings
# Vector Store (Local/Gratuita)
from langchain_community.vectorstores import FAISS
# Cadeias (Chains)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Carrega as variáveis de ambiente (sua GOOGLE_API_KEY)
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# --- Constantes ---
# O NOME DA PASTA QUE VOCÊ FEZ UPLOAD PARA O GITHUB
FAISS_INDEX_PATH = "faiss_index_projeto" 

# --- Funções Cacheadas (A Mágica do Streamlit) ---

# O @st.cache_resource "guarda" o modelo na memória do Streamlit.
# Isso garante que só vamos baixar/carregar o modelo UMA VEZ.
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

# Cacheia o carregamento do LLM
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

# Cacheia o carregamento do índice FAISS (que está no GitHub)
@st.cache_resource
def load_faiss_index(embeddings_model):
    print("Carregando índice FAISS local...")
    # Carrega o índice da pasta (que você upou pro GitHub)
    db = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings_model, 
        allow_dangerous_deserialization=True # Necessário para carregar do disco
    )
    print("Índice FAISS carregado.")
    return db

# Cacheia a criação da cadeia RAG
@st.cache_resource
def get_rag_chain(_llm, _retriever):
    print("Criando a cadeia RAG...")
    # Template do Prompt
    prompt_template = """
    Você é um assistente especialista. Responda a pergunta *apenas* com base no contexto.
    Contexto: {context}
    Pergunta: {input}
    Resposta:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Cria as duas cadeias
    document_chain = create_stuff_documents_chain(_llm, prompt)
    retrieval_chain = create_retrieval_chain(_retriever, document_chain)
    print("Cadeia RAG pronta.")
    return retrieval_chain

# --- Interface do Streamlit ---

st.set_page_config(page_title="Chat com Documentos", layout="wide")
st.title("📄 Chatbot com Documentos (Usando Gemini)")

# Garante que a API Key foi configurada
if not google_api_key:
    st.error("GOOGLE_API_KEY não encontrada! Configure-a nos 'Secrets' do Streamlit.")
else:
    try:
        # --- Carregamento dos Modelos (via cache) ---
        embeddings = get_embeddings_model()
        llm = get_llm()
        
        # Carrega o índice FAISS e o transforma em um "buscador"
        db = load_faiss_index(embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 4})
        
        # Carrega a cadeia RAG
        chain = get_rag_chain(llm, retriever)

        # --- Interface de Chat ---
        st.write("O índice está carregado. Faça sua pergunta sobre o documento.")
        
        # Input do usuário
        user_question = st.text_input("Sua pergunta:")

        if user_question:
            # Mostra um "spinner" enquanto pensa
            with st.spinner("Pensando... (Consultando o Gemini e o índice)"):
                # Invoca a cadeia
                response = chain.invoke({"input": user_question})
                
                # Mostra a resposta
                st.subheader("Resposta:")
                st.write(response["answer"])

    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar os componentes: {e}")
        st.error("Verifique se a pasta 'faiss_index_projeto' existe no seu repositório.")