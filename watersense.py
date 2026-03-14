import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="WaterSense QA Bot", page_icon="💧", layout="centered")

st.markdown("""
<style>
.main { background-color: #e8f4f8; }
.stApp { background: linear-gradient(135deg, #e0f7fa, #b2ebf2, #e0f2f1); }
.welcome-box {
    background-color: #ffffff;
    border-left: 6px solid #0077b6;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("💧 WaterSense QA Bot")
st.markdown("### *Your Intelligent Water Quality Assistant*")
st.divider()

st.markdown("""
<div class="welcome-box">
    <h4>👋 Welcome to WaterSense QA Bot!</h4>
    <p>Thank you for choosing WaterSense — <b>you have made the right decision!</b> 🎉</p>
    <p>This AI-powered assistant is designed to help you understand water quality parameters, 
    WHO guidelines, treatment methods, and research findings — instantly and accurately.</p>
    <p>💡 <b>Try asking:</b></p>
    <ul>
        <li>What is the safe pH range for drinking water?</li>
        <li>What are WHO guidelines for nitrates?</li>
        <li>How is turbidity measured?</li>
        <li>What are the health effects of heavy metals in water?</li>
        <li>What are common water treatment methods?</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.divider()

@st.cache_resource
def load_vectorstore():
    docs_path = "docs"
    all_docs = []
    for file in os.listdir(docs_path):
        filepath = os.path.join(docs_path, file)
        if file.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            all_docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

@st.cache_resource
def get_qa_chain(_vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})
    return llm, retriever

vectorstore = load_vectorstore()
llm, retriever = get_qa_chain(vectorstore)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("💧 Ask me anything about water quality..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            docs = retriever.invoke(prompt)
            context = "\n\n".join([doc.page_content for doc in docs])
            full_prompt = f"You are a water quality expert. Answer based on this context:\n\n{context}\n\nQuestion: {prompt}"
            response = llm.invoke(full_prompt)
            answer = response.content
        st.markdown(answer)
        with st.expander("📚 View Sources"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                st.markdown(f"> {doc.page_content[:200]}...")
    st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.markdown("<p style='text-align:center; color:gray;'>© 2026 WaterSense QA Bot | Powered by Google Gemini & LangChain</p>", unsafe_allow_html=True)