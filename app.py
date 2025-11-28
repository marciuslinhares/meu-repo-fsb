import streamlit as st
import os
import shutil

# --- CONFIGURAÇÃO DA PÁGINA (Deve ser o primeiro comando Streamlit) ---
st.set_page_config(
    page_title="FSB | Knowledge Hub",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SEGURANÇA E SETUP DE API ---
# Em produção (Streamlit Cloud), a chave vem de st.secrets.
# Localmente, você pode ter um arquivo .streamlit/secrets.toml
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("⚠️ Erro: Chave de API não configurada nos Secrets.")
    st.stop()

# --- IMPORTAÇÕES ---
# Importações colocadas após o set_page_config para evitar erros
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CSS VISUAL FSB ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    
    .stApp { background-color: #F9FBF9; }
    header[data-testid="stHeader"] { background-color: #F9FBF9; }

    /* Sidebar Verde FSB */
    [data-testid="stSidebar"] { background-color: #16241E; border-right: 2px solid #8CC63F; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #FFFFFF !important; }

    /* Chat Bubbles */
    .stChatMessage.st-emotion-cache-1c7y2kd { background-color: #16241E; color: white; border-radius: 15px 15px 0px 15px; }
    .stChatMessage.st-emotion-cache-1c7y2kd p { color: #FFFFFF !important; }
    .stChatMessage.st-emotion-cache-4oy321 { background-color: #FFFFFF; border-left: 4px solid #8CC63F; border-radius: 4px 15px 15px 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .stChatMessage.st-emotion-cache-4oy321 p { color: #333333 !important; }

    /* Header Customizado */
    .fsb-banner { display: flex; align-items: center; background-color: white; padding: 1rem 0; border-bottom: 3px solid #8CC63F; margin-bottom: 2rem; }
    .fsb-logo-text { font-size: 3rem; font-weight: 800; color: #16241E; margin-right: 2px; line-height: 1; }
    .fsb-dot { font-size: 3rem; font-weight: 800; color: #8CC63F; line-height: 1; }
    .fsb-subtitle { margin-left: 20px; font-size: 1.1rem; color: #666; border-left: 1px solid #ccc; padding-left: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="fsb-banner">
    <span class="fsb-logo-text">fsb</span><span class="fsb-dot">.</span>
    <span class="fsb-subtitle">Knowledge Hub AI</span>
</div>
""", unsafe_allow_html=True)

# --- FUNÇÃO DE CARREGAMENTO INTELIGENTE (ORQUESTRADOR) ---
@st.cache_resource(show_spinner=False)
def load_and_process_documents():
    base_folder = "base_conhecimento"
    
    if not os.path.exists(base_folder):
        # Cria a pasta apenas para evitar erro, mas avisa o usuário
        os.makedirs(base_folder)
        return None, 0

    documents = []
    total_files = 0
    
    # Varre todas as subpastas procurando PDFs e TXTs
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                    total_files += 1
                elif file.endswith(".txt"):
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                    total_files += 1
            except Exception as e:
                print(f"Erro ao ler {file}: {e}")

    if not documents:
        return None, 0

    # Processamento e Vector Store
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    
    # Embeddings do Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Cria o Vectorstore em memória para esta sessão
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    return vectorstore, total_files

# --- CARREGAMENTO INICIAL ---
with st.spinner("Inicializando base de conhecimento..."):
    vectorstore, file_count = load_and_process_documents()

# --- SIDEBAR DINÂMICA ---
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🗂️ BASE DE DADOS")
    st.markdown("---")
    
    if file_count > 0:
        st.success(f"**{file_count} documentos** indexados.")
        st.caption("Sistema operante.")
    else:
        st.warning("Nenhum arquivo encontrado.")
        st.info("Certifique-se de que a pasta 'base_conhecimento' existe no repositório GitHub e contém arquivos.")

    st.caption("FSB Comunicação © 2025")

# --- LÓGICA DE RESPOSTA ---
def get_answer_from_vectorstore(query):
    if not vectorstore:
        return "⚠️ A pasta `base_conhecimento` está vazia ou não foi carregada corretamente."

    # Retrieval (Busca os 6 trechos mais relevantes)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    # Ajustado para gemini-1.5-flash (Versão estável atual)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
    
    template = """Você é o Assistente Corporativo da FSB.
    Você tem acesso a múltiplos documentos internos.
    
    Use o contexto abaixo para responder de forma profissional.
    
    CONTEXTO RECUPERADO:
    {context}
    
    PERGUNTA: {question}
    
    Resposta:"""
    
    prompt = PromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(query)

# --- CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Tenho acesso a todas as pastas de conhecimento da FSB. O que deseja saber?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ex: Qual a política de PLR? ou Quais os projetos atuais?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Consultando documentos..."):
            try:
                response = get_answer_from_vectorstore(query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Erro ao processar: {e}")