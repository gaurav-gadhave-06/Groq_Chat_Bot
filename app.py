import streamlit as st
import os
import time
from dotenv import load_dotenv

# ✅ Updated Import for OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings  

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# ✅ Load the Groq API key securely
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Groq API key is missing! Please set GROQ_API_KEY in your environment variables.")
    st.stop()

# Streamlit session state initialization
if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="tinyllama")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit UI
st.title("ChatGroq Demo")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Question: {input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User Input
user_input = st.text_input("Input your prompt here")

if user_input:
    start_time = time.process_time()
    response = retrieval_chain.invoke({"input": user_input})
    response_time = time.process_time() - start_time

    print("Response time:", response_time)
    st.write(response.get("answer", "No answer found."))

    # Display relevant documents
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response.get("context", [])):
            st.write(doc.page_content)
            st.write("--------------------------------")
