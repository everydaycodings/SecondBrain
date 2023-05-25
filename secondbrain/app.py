# main.py
import os
import tempfile

import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from supabase import Client, create_client
from helpers.source_embedding import load_pdf, load_embedding_model, retriver, source_docs
from helpers.add_knowledge import extract_content


# Set the theme
st.set_page_config(
    page_title="SecondBrain",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("SecondBrain 🧠")
st.markdown("Store your knowledge in a vector store and query it with OpenAI's GPT-3/4.")

st.markdown("---\n\n")

user_choice = st.sidebar.selectbox("Select Your Choice: ", options=['Add Knowledge', 'Chat with your Brain', 'Forget', "Explore"])


# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state['model'] = "gpt-3.5-turbo"
if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 0.0
if 'chunk_size' not in st.session_state:
    st.session_state['chunk_size'] = 500
if 'chunk_overlap' not in st.session_state:
    st.session_state['chunk_overlap'] = 0
if 'max_tokens' not in st.session_state:
    st.session_state['max_tokens'] = 256

# Create a radio button for user to choose between adding knowledge or asking a question


if user_choice == 'Add Knowledge':
    
    st.sidebar.title("Configuration")
    st.sidebar.markdown(
        "Choose your chunk size and overlap for adding knowledge.")
    st.session_state['chunk_size'] = st.sidebar.slider(
        "Select Chunk Size", 100, 1000, st.session_state['chunk_size'], 50)
    st.session_state['chunk_overlap'] = st.sidebar.slider(
        "Select Chunk Overlap", 0, 500, st.session_state['chunk_overlap'], 10)
    
    knowledge_sources = st.file_uploader(label="Upload Multiple PDFs: ", accept_multiple_files=True)
    if st.button("Add To Database"):
        files = extract_content(knowledge_sources)
        for f in files:
            st.text(f)


if user_choice == "Source Embedding":

    st.sidebar.title("Configuration")
    st.sidebar.markdown(
        "Choose your chunk size and overlap for adding knowledge.")
    st.session_state['chunk_size'] = st.sidebar.slider(
        "Select Chunk Size", 100, 1000, st.session_state['chunk_size'], 50)
    st.session_state['chunk_overlap'] = st.sidebar.slider(
        "Select Chunk Overlap", 0, 500, st.session_state['chunk_overlap'], 10)
    
    col1, col2 = st.columns(2)

    with col1:
        pdf_file = st.file_uploader("Upload The PDF: ")
        search_arg = st.number_input("Number of Closest Relevent Search: ", min_value=1, step=1, value=3)
    with col2:
        model_name = st.selectbox(label="Select Your Source Embedding Model: ", options=["hkunlp/instructor-xl"])
        st.text(" ")
        st.text(" ")
        device = st.selectbox(label="Select Your Device: ", options=["cuda", "cpu"])

    
    if st.button("Load Model"):
        with st.spinner("Model Loading..."):
            pdf_pages = load_pdf(pdf_file)
            embeddingings, condition = load_embedding_model(model_name, device)
            retrive = retriver(embeddingings, pdf_pages, search_arg)
        
        prompt = st.text_input("Enter Your Prompt: ", value="Explain Proof-Of-Work?")
        docs_result =source_docs(retrive, prompt)
        st.text(docs_result)