import os 
from langchain_community.document_loaders import DirectoryLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS 

DATA_PATH = "data" 

FAISS_INDEX_PATH = "faiss_index_projeto" 

def get_embeddings_model():
    print("Carregando modelo de embedding local (HuggingFace)...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    print("Modelo de embedding carregado.")
    return embeddings

def criar_index():
    if not os.path.exists(DATA_PATH):
        print(f"Erro: Pasta de dados '{DATA_PATH}' não encontrada.")
        print("Certifique-se de que a pasta 'data' com seus arquivos .txt está no mesmo local do indexador.py")
        return

    if os.path.exists(FAISS_INDEX_PATH):
        print(f"A pasta de índice '{FAISS_INDEX_PATH}' já existe. Pulando a criação.")
        return
    
    try:
        embeddings = get_embeddings_model()
        
        print(f"Carregando documentos da pasta '{DATA_PATH}'...")
        loader = DirectoryLoader(
            DATA_PATH,
            glob="**/*.txt",
            show_progress=True,
            use_multithreading=True
        )
        
        documentos = loader.load()

        if not documentos:
            print(f"Nenhum arquivo .txt encontrado em '{DATA_PATH}'. Verifique a pasta.")
            return

        print(f"Total de {len(documentos)} documentos carregados da pasta 'data'.")

        print("Dividindo o texto em 'chunks' (pedaços)...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documentos)

        print(f"Criando embeddings para {len(docs)} chunks e salvando no índice FAISS...")
        db = FAISS.from_documents(docs, embeddings)
        
        db.save_local(FAISS_INDEX_PATH)
        print("\n--- SUCESSO! ---")
        print(f"A pasta '{FAISS_INDEX_PATH}' foi criada com sucesso.")
        print("Agora, envie esta nova pasta para o seu repositório no GitHub.")
    
    except Exception as e:
        print(f"\nOcorreu um erro durante a indexação: {e}")

if __name__ == "__main__":
    criar_index()