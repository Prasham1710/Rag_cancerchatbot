import streamlit as st
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

# ------------------- UI Styling -------------------
st.markdown("""
<style>
    .main, .block-container {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stTextInput textarea, .stTextArea textarea {
        color: #ffffff !important;
        background-color: #2d2d2d !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: #ffffff !important;
        background-color: #3d3d3d !important;
    }
    .stSelectbox svg {
        fill: #ffffff !important;
    }
    .stSelectbox option, div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
    .stMarkdown, .stMarkdown p, .stChatMessage {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Cancer Buddy")
st.caption("🚀 Your Cancer research assistant powered by RAG")

# ------------------- LLM and Vector Store -------------------
llm_engine = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.3
)

@st.cache_resource
def load_vectorstore():
    loader1 = PyPDFLoader("cancer_docs/cancer.pdf")
    loader2 = PyPDFLoader("cancer_docs/management.pdf")
    documents = loader1.load() + loader2.load()
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

with st.spinner("🔍 Loading cancer documents into memory..."):
    retriever = load_vectorstore()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm_engine,
    retriever=retriever,
    return_source_documents=True
)

if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "ai", "content": "Hi! I'm CancerBuddy. How can I help you today? 💻"}
    ]

# ------------------- Format Final Response -------------------
def format_final_response(raw_response, sources, query):
    cleaned_response = re.sub(r"<\/?think.?>|\[thinking.?\]", "", raw_response, flags=re.IGNORECASE).strip()

    # Check if a conclusion is already present
    conclusion_phrases = [
        "in conclusion", "overall", "to summarize", "therefore",
        "it's recommended", "it is safe", "not advised", "we can conclude"
    ]
    if not any(phrase in cleaned_response.lower() for phrase in conclusion_phrases):
        cleaned_response += "\n\n**Conclusion:** Based on the current evidence, this topic should be considered as part of a broader, balanced approach to cancer care. For personalized recommendations, always consult a healthcare provider."

    # Add source references
    if sources:
        cleaned_response += "\n\n**Sources:**"
        for doc in sources:
            filename = doc.metadata.get("source", "PDF")
            page = doc.metadata.get("page", "Unknown")
            cleaned_response += f"\n- Page {page} of `{filename}`"

    return cleaned_response

# ------------------- Chat Interface -------------------
user_query = st.chat_input("Type your question here...")

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})

    result = qa_chain({"query": user_query})
    raw_response = result["result"]
    sources = result.get("source_documents", [])

    final_response = format_final_response(raw_response, sources, user_query)
    st.session_state.message_log.append({"role": "ai", "content": final_response})
    st.rerun()

# ------------------- Display Chat -------------------
with st.container():
    for msg in st.session_state.message_log:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
