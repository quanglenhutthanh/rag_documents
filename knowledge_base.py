import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
import docx2txt
from PyPDF2 import PdfReader

def get_vectordb():
    load_dotenv()
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
        model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        api_version="2024-07-01-preview"
    )
    vectordb = Chroma(persist_directory=".chromadb", embedding_function=embeddings)
    return vectordb

def parse_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def parse_txt(file) -> str:
    return file.read().decode("utf-8")

def parse_docx(file) -> str:
    temp_path = "temp.docx"
    with open(temp_path, "wb") as f:
        f.write(file.read())
    text = docx2txt.process(temp_path)
    os.remove(temp_path)
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def knowledge_base_tab():
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )
    if uploaded_files:
        all_chunks = []
        chunk_metadatas = []
        file_names = []
        for file in uploaded_files:
            if file.type == "application/pdf":
                text = parse_pdf(file)
            elif file.type == "text/plain":
                text = parse_txt(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = parse_docx(file)
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            chunk_metadatas.extend([{"source": file.name}] * len(chunks))
            file_names.append(file.name)
        st.write(f"Total new chunks: {len(all_chunks)}")
        if st.button("Embed and Store in ChromaDB"):
            vectordb = get_vectordb()
            vectordb.add_texts(all_chunks, metadatas=chunk_metadatas)
            vectordb.persist()
            st.success("Documents embedded and stored!")
            if 'file_list' not in st.session_state:
                st.session_state['file_list'] = []
            st.session_state['file_list'].extend(file_names)
    st.subheader("Embedded Files")
    if st.session_state.get('file_list'):
        for fname in set(st.session_state['file_list']):
            st.markdown(f"- {fname}")
    else:
        st.info("No files embedded yet.")
    vectordb = get_vectordb()
    stats = vectordb._collection.count()
    st.session_state['db_stats'] = {"chunks": stats}
    st.subheader("DB Stats")
    st.write(st.session_state['db_stats'])