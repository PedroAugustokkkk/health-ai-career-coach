# 🩺 Sanar AI Career Coach (Protótipo)

> Um chatbot de RAG (Retrieval-Augmented Generation) que transforma o conteúdo estático do blog da Sanar em um assistente de carreira interativo.

Este projeto é um protótipo de ferramenta de IA desenhado para a Sanar, demonstrando como alavancar o conteúdo existente para aumentar o engajamento e consolidar a autoridade da marca.
Você pode testar agora, acessando a URL: https://health-ai-career-coach.streamlit.app

## 🎯 O Problema

Estudantes de medicina têm inúmeras dúvidas complexas sobre carreira, residência e especialidades. As respostas para essas perguntas já existem no blog da Sanar, mas estão dispersas por centenas de artigos. Encontrar a informação exata é um processo manual e demorado para o usuário.

## 💡 A Solução

Um "Assistente de Carreira" que utiliza uma arquitetura RAG. O sistema indexa todos os artigos relevantes do blog e usa um LLM (Google Gemini) para responder perguntas em linguagem natural.

O estudante pode perguntar, "Quais as residências mais concorridas em São Paulo para cardiologia?", e o bot irá sintetizar uma resposta precisa, baseada **exclusivamente** no conteúdo oficial do blog, citando as fontes.

**Valor para o Negócio:**
* **Aumento de Engajamento:** Transforma leitores passivos em usuários ativos.
* **Centralização da Informação:** Torna-se a ferramenta "go-to" para dúvidas de carreira.
* **Autoridade:** Reforça a imagem da Sanar como a fonte definitiva de conhecimento.

## ✨ Funcionalidades Principais

* **Chat com IA (RAG):** Respostas geradas pelo Google Gemini (Flash) com base no contexto injetado.
* **Indexação de Conteúdo:** Lê e vetoriza todos os artigos (`.txt`) colocados na pasta `/data`.
* **"Grounded" (Aterrado):** O prompt do sistema instrui o LLM a se ater estritamente aos fatos encontrados nos artigos, prevenindo "alucinações" ou informações incorretas.

## 🛠️ Stack de Tecnologia

* **Frontend:** Streamlit
* **Orquestração RAG:** LangChain
* **LLM (Geração):** Google Gemini 2.5 Flash (via API)
* **Embeddings (Vetorização):** Hugging Face `all-MiniLM-L6-v2` (Local, 100% gratuito)
* **Vector Store (Busca):** FAISS-CPU (em memória)

## 🚀 Como Executar Localmente

1.  Clone o repositório:
    ```bash
    git clone [https://github.com/seu-usuario/sanar-ai-coach.git](https://github.com/seu-usuario/sanar-ai-coach.git)
    cd sanar-ai-coach
    ```

2.  Crie e ative um ambiente virtual:
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

4.  Popule a base de conhecimento:
    * Adicione seus artigos em formato `.txt` dentro da pasta `/data`.

5.  Configure suas chaves de API (veja abaixo).

6.  Execute a aplicação:
    ```bash
    streamlit run app.py
    ```

## 🔑 Configuração

Crie um arquivo `.env` na raiz do projeto e adicione sua chave da API do Google:

```plaintext
GOOGLE_API_KEY="sua-chave-secreta-do-google-aqui"
