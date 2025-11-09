# -*- coding: utf-8 -*-

# --- 1. Importações ---
import streamlit as st
import os
from dotenv import load_dotenv
import langchain


# Importações específicas do Google Generative AI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Componentes LangChain Core (Para Prompts)
from langchain_core.prompts import ChatPromptTemplate # <- MUDANÇA AQUI

# Componentes LangChain (Para Chains)
# ISSO ESTÁ CERTO (COLA ISSO):
from langchain import create_stuff_documents_chain
from langchain import create_retrieval_chain

# Componentes LangChain Community (I/O e Armazenamento)
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Componentes de processamento de texto (Pacote separado)
from langchain_text_splitters import RecursiveCharacterTextSplitter # <- MUDANÇA AQUI

# --- 2. Configuração e Variáveis de Ambiente ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- 3. Carregamento e Indexação de Documentos ---

@st.cache_resource 
def load_and_index_documents():
    """
    Carrega documentos .txt, os divide em chunks, gera embeddings (usando Google)
    e cria um VectorStore FAISS em memória.
    """
    try:
        loader = DirectoryLoader(
            "./data",
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()

        if not documents:
            st.error("Diretório 'data' não encontrado ou vazio. Nenhum documento .txt foi carregado.")
            return None

        # O TextSplitter agora vem do 'langchain_text_splitters'
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        st.error(f"Erro durante a indexação (verifique sua GOOGLE_API_KEY): {e}")
        return None

# --- 4. Configuração da Chain de RAG (Retrieval-Augmented Generation) ---

def setup_rag_chain(retriever):
    """
    Configura a chain de RAG com o LLM do Google (Gemini).
    """
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    # O ChatPromptTemplate agora vem do 'langchain_core.prompts'
    prompt_template = """
    Você é o "Sanar AI Career Coach", um assistente especialista focado em carreira médica,
    baseado exclusivamente nos dados da Sanar.
    
    Responda à pergunta do usuário utilizando *apenas* o contexto fornecido abaixo.
    Se a resposta não estiver contida no contexto, informe:
    "Desculpe, eu não tenho informações sobre isso nos artigos da Sanar que consultei."

    Contexto:
    {context}

    Pergunta:
    {input}

    Resposta:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

# --- 5. Interface Streamlit (UI) ---

def main():
    st.set_page_config(page_title="Sanar AI Career Coach (Gemini)", page_icon="🩺")
    st.title("🩺 Sanar AI Career Coach")
    st.write("Assistente de carreira médica (Powered by Google Gemini)")

    try:
        retriever = load_and_index_documents()

        if retriever:
            rag_chain = setup_rag_chain(retriever)

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if user_query := st.chat_input("Qual sua dúvida sobre residência?"):
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.spinner("Analisando artigos da Sanar (via Gemini)..."):
                    response = rag_chain.invoke({"input": user_query})
                    ai_response = response["answer"]

                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                with st.chat_message("assistant"):
                    st.markdown(ai_response)

    except Exception as e:
        st.error(f"Ocorreu um erro crítico na aplicação: {e}")
        st.info("Verifique se a GOOGLE_API_KEY está correta no .env e reinicie.")

# --- Ponto de Entrada ---
if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        st.error("Chave GOOGLE_API_KEY não encontrada.")
        st.info("Por favor, configure sua chave no arquivo .env e reinicie a aplicação.")
    else:
        main()