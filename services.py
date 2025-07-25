import streamlit as st
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.vectorstores import Chroma



def get_vectordb():
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=st.secrets["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
        api_key=st.secrets["AZURE_OPENAI_EMBEDDING_API_KEY"],
        model=st.secrets["AZURE_OPENAI_EMBEDDING_MODEL"],
        chunk_size=1000
    )
    return Chroma(persist_directory=".chromadb", embedding_function=embeddings)

def get_azure_llm():
    return AzureChatOpenAI(
        azure_endpoint=st.secrets["AZURE_OPENAI_LLM_ENDPOINT"],
        api_key=st.secrets["AZURE_OPENAI_LLM_API_KEY"],
        model=st.secrets["AZURE_OPENAI_LLM_MODEL"],
        api_version="2024-07-01-preview"
    )
