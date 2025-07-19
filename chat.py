import streamlit as st
import re
import os
from dotenv import load_dotenv
from typing import TypedDict, List, Tuple
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chat_models import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from services import get_azure_llm, get_vectordb

load_dotenv()

class ChatState(TypedDict, total=False):
    question: str
    chat_history: List[Tuple[str, str]]
    context_docs: List[str]
    answer: str
    next_step: str

def retrieve_docs(state:ChatState) -> ChatState:
    retriever = get_vectordb().as_retriever(search_kwargs={"k": 4}) 
    docs = retriever.get_relevant_documents(state["question"])
    return {**state, "context_docs": docs}
def generate_answer(state: ChatState) -> ChatState:
    llm = get_azure_llm()
    question = state["question"]
    chat_history = state["chat_history"]
    docs = state.get("context_docs", [])

    # If context_docs are available, use them; else fall back to other state (e.g. compare_price/search_news sets `answer`)
    if docs:
        context = "\n".join([doc.page_content for doc in docs])
        response = llm.invoke(f"Context: {context}\n\nQuestion: {question}")
        return {**state, "answer": response.content}
    elif "answer" in state:
        # Already has answer (from compare_price or search_news)
        return state
    else:
        # No context or answer; fallback
        return {**state, "answer": "I'm not sure how to help with that right now."}
# def generate_answer(state: ChatState) -> ChatState:
#     llm = get_azure_llm()
#     question = state["question"]
#     chat_history = state["chat_history"]
#     docs = state["context_docs"]

#     context = "\n".join([doc.page_content for doc in docs])
#     response = llm.invoke(f"Context: {context}\n\nQuestion: {question}")
    
#     return {**state, "answer": response.content}

def compare_price(state: ChatState) -> ChatState:
    question = state["question"].lower()
    product = "car insurance" if "insurance" in question else "car battery"
    
    price_data = {
        "car insurance": [
            ("Vendor A", "$500/year"),
            ("Vendor B", "$450/year"),
            ("Vendor C", "$475/year")
        ],
        "car battery": [
            ("Vendor A", "$150"),
            ("Vendor B", "$145"),
            ("Vendor C", "$160")
        ]
    }

    price_list = price_data.get(product, [])
    response_lines = [f"{vendor}: {price}" for vendor, price in price_list]
    response = f"Price comparison for **{product}**:\n\n" + "\n".join(response_lines)

    return {**state, "answer": response}

def search_news(state: ChatState) -> ChatState:
    topic = state["question"].lower()
    
    mock_articles = [
        f"🚘 Breaking: Electric vehicle sales surge 25% in Q2 2025.",
        f"🔧 Update: Toyota announces new hybrid models for 2026.",
        f"🛠️ Analysis: How EV maintenance compares with gas-powered cars."
    ]
    
    response = f"Here are some recent news headlines related to **{topic}**:\n\n"
    response += "\n".join(f"- {article}" for article in mock_articles)

    return {**state, "answer": response}

intent_prompt = PromptTemplate.from_template(
    """You are an intent classifier.
        Classify the user's question into one of the following:
        - retrieve_docs
        - compare_price
        - search_news

        User question: {question}

        Respond with only the intent label."""
)
intent_classifier: RunnableSequence = (
    intent_prompt 
    | get_azure_llm() 
    | StrOutputParser()
)
def route_user_input(state: ChatState) -> ChatState:
    intent = intent_classifier.invoke({"question": state["question"]}).strip()
    if intent not in ["retrieve_docs", "compare_price", "search_news"]:
        intent = "retrieve_docs"
    return {**state, "next_step": intent}

graph = StateGraph(ChatState)

# Add all logic nodes
graph.add_node("router", route_user_input)
graph.add_node("retrieve_docs", retrieve_docs)
graph.add_node("compare_price", compare_price)
graph.add_node("search_news", search_news)
graph.add_node("generate_answer", generate_answer)

# Set the entry point
graph.set_entry_point("router")

# Conditional routing from router
graph.add_conditional_edges("router", lambda state: state["next_step"], {
    "retrieve_docs": "retrieve_docs",
    "compare_price": "compare_price",
    "search_news": "search_news"
})

# Continue each path to generate_answer, then end
graph.add_edge("retrieve_docs", "generate_answer")
graph.add_edge("compare_price", "generate_answer")
graph.add_edge("search_news", "generate_answer")
graph.add_edge("generate_answer", END)

# Compile the workflow
workflow = graph.compile()

def chat_tab():
    st.header("Chat with your documents")
    vectordb = get_vectordb()
    USER_BUBBLE = "background-color:#e6f7ff;padding:12px 16px;border-radius:12px 12px 12px 2px;margin-bottom:4px;display:inline-block;max-width:80%;"
    AI_BUBBLE = "background-color:#f6f6f6;padding:12px 16px;border-radius:12px 12px 2px 12px;margin-bottom:16px;display:inline-block;max-width:80%;"
    def format_ai_answer(ans: str) -> str:
        ans = re.sub(r"Step (\d+):", r"\n\1. ", ans)
        ans = re.sub(r"\.\s+", ".\n\n", ans)
        return ans.strip()
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    for user_msg, ai_msg in st.session_state['chat_history']:
        st.markdown(f'<div style=\"{USER_BUBBLE}\"><b>You:</b> {user_msg}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style=\"{AI_BUBBLE}\"><b>AI:</b> </div>', unsafe_allow_html=True)
        st.markdown(format_ai_answer(ai_msg), unsafe_allow_html=False)
        st.markdown("<hr style='border:0;border-top:1px solid #eee;margin:8px 0;'>", unsafe_allow_html=True)
    
    # Generate answer before chat input so spinner is above input
    if st.session_state['chat_history'] and st.session_state['chat_history'][-1][1] == "...":
        user_msg = st.session_state['chat_history'][-1][0]
    
        with st.spinner("Generating answer..."):
            result = workflow.invoke({
                "question": user_msg,
                "chat_history": st.session_state['chat_history'][:-1]
                
            })
            answer = result["answer"]
        st.session_state['chat_history'][-1] = (user_msg, answer)
        st.rerun()
    user_input = st.chat_input("Ask a question about your documents...")
    if user_input:
        st.session_state['chat_history'].append((user_input, "..."))
        st.rerun()
    if st.button("Clear Chat History"):
        st.session_state['chat_history'] = []
        st.rerun()


# For compare_price:
# "Can you compare car insurance prices?"
# "What's the cheapest battery for my car?"

# For search_news:
# "Show me news about electric vehicles"
# "What’s new in the automotive industry?"

