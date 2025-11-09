# --- Imports necessários para a indexação ---

# Para interagir com o sistema operacional (verificar pastas)
import os 
# !! MUDANÇA IMPORTANTE !!
# Usaremos o DirectoryLoader para carregar múltiplos arquivos de um diretório
from langchain_community.document_loaders import DirectoryLoader
# Para dividir o texto em pedaços (chunks)
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter 
# Para o modelo de embedding local (gratuito)
from langchain_huggingface import HuggingFaceEmbeddings 
# Para o banco de dados vetorial
from langchain_community.vectorstores import FAISS 

# --- 2. Configuração dos Nomes ---

# !! MUDANÇA IMPORTANTE !!
# O caminho para a pasta que contém seus artigos .txt
DATA_PATH = "data" 

# Nome da pasta onde o índice será salvo
# (O seu app.py no Streamlit vai ler desta pasta)
FAISS_INDEX_PATH = "faiss_index_projeto" 

# --- 3. Função para carregar o modelo de embedding ---
# (Esta função está correta e não muda)
def get_embeddings_model():
    # Imprime um status para sabermos o que está acontecendo
    print("Carregando modelo de embedding local (HuggingFace)...")
    # Define o nome do modelo (leve e popular)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # Define onde o modelo vai rodar ('cpu' ou 'cuda' se tiver GPU)
    model_kwargs = {'device': 'cpu'}
    # Cria a instância do modelo de embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    # Imprime um status de sucesso
    print("Modelo de embedding carregado.")
    # Retorna o modelo pronto para uso
    return embeddings

# --- 4. Função principal de indexação (ATUALIZADA) ---
def criar_index():
    # Verifica se a pasta 'data' (com seus .txt) existe
    if not os.path.exists(DATA_PATH):
        # Informa o usuário se a pasta não for encontrada
        print(f"Erro: Pasta de dados '{DATA_PATH}' não encontrada.")
        print("Certifique-se de que a pasta 'data' com seus arquivos .txt está no mesmo local do indexador.py")
        # Para a execução
        return

    # Verifica se o índice já foi criado para não refazer o trabalho
    if os.path.exists(FAISS_INDEX_PATH):
        # Informa que o índice já existe
        print(f"A pasta de índice '{FAISS_INDEX_PATH}' já existe. Pulando a criação.")
        # Para a execução
        return
    
    # Tenta executar o bloco de indexação
    try:
        # Carrega o modelo de embedding (só na primeira vez vai baixar)
        embeddings = get_embeddings_model()
        
        # !! MUDANÇA IMPORTANTE !!
        # Informa o usuário sobre o carregamento da pasta
        print(f"Carregando documentos da pasta '{DATA_PATH}'...")
        # Configura o loader para procurar todos os arquivos .txt na pasta 'data'
        loader = DirectoryLoader(
            DATA_PATH,                  # O caminho da pasta
            glob="**/*.txt",            # O padrão de arquivo (todos os .txt)
            show_progress=True,         # Mostra uma barra de progresso
            use_multithreading=True     # Usa threads para carregar mais rápido
        )
        
        # Carrega os documentos
        documentos = loader.load()

        # Verifica se algum documento foi carregado
        if not documentos:
            # Avisa se a pasta 'data' estiver vazia
            print(f"Nenhum arquivo .txt encontrado em '{DATA_PATH}'. Verifique a pasta.")
            # Para a execução
            return

        # Informa quantos arquivos foram lidos
        print(f"Total de {len(documentos)} documentos carregados da pasta 'data'.")

        # Informa sobre a divisão do texto
        print("Dividindo o texto em 'chunks' (pedaços)...")
        # Cria o objeto para dividir o texto (chunks de 1000 caracteres)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # Executa a divisão dos documentos carregados
        docs = text_splitter.split_documents(documentos)

        # Informa sobre a criação dos vetores (embeddings)
        print(f"Criando embeddings para {len(docs)} chunks e salvando no índice FAISS...")
        # Esta é a etapa principal: usa o modelo local para criar os vetores
        db = FAISS.from_documents(docs, embeddings)
        
        # Salva o banco de dados vetorial localmente na pasta definida
        db.save_local(FAISS_INDEX_PATH)
        
        # Imprime a mensagem final de sucesso
        print("\n--- SUCESSO! ---")
        print(f"A pasta '{FAISS_INDEX_PATH}' foi criada com sucesso.")
        print("Agora, envie esta nova pasta para o seu repositório no GitHub.")
    
    # Captura qualquer erro que possa acontecer
    except Exception as e:
        # Imprime o erro
        print(f"\nOcorreu um erro durante a indexação: {e}")

# --- 5. Execução do Script ---
# (Esta é a linha que "chama" a função principal quando você roda 'python indexador.py')
if __name__ == "__main__":
    # Chama a função 'criar_index'
    criar_index()