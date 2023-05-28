# main.py
import os
import tempfile
import random
import time
import streamlit as st
from streamlit_chat import message as st_message

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from supabase import Client, create_client
from helpers.add_knowledge import AddKnowledge
from helpers.source_embedding import ChatSourceEmbedding
from helpers.utils import list_folder_name, get_model_path
from helpers.wandering_brain import WanderingBrain

# Set the theme
st.set_page_config(
    page_title="SecondBrain",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("SecondBrain 🧠")
st.markdown("Store your knowledge in a vector store and query it with OpenAI's GPT-3/4.")

st.markdown("---\n\n")

user_choice = st.sidebar.selectbox("Select Your Choice: ", options=['Add Knowledge', "Chat Source Embedding", "Wandering Brain", 'Chat with your Brain', 'Forget', "Explore"])


if user_choice == 'Add Knowledge':
    
    st.sidebar.title("Configuration")

    with st.sidebar.expander("Configuration"):
        model_name = st.selectbox(label="Select Your Source Embedding Model: ", options=["hkunlp/instructor-xl"])
        device = st.selectbox(label="Select Your Device: ", options=["cuda", "cpu"])
        chunk_size = st.slider(label="Select Chunk Size", min_value=100, max_value=1000, step=1, value=500)
        chunk_overlap = st.slider(label="Select Chunk Overlap", min_value=0, max_value=500, step=1, value=10)
        embedding_storing_dir = st.text_input(label="Enter the name of the Database: ", value="db")
        st.text(" ")
        st.text(" ")

    knowledge_sources = st.file_uploader(label="Upload Multiple PDFs: ", accept_multiple_files=True)

        
    
    if st.button("Add To Database"):
        
        with st.spinner("Adding.."):
            add_knowledge = AddKnowledge()
            files = add_knowledge.extract_content(knowledge_sources, chunk_size, chunk_overlap)
            st.info("Content Extracted")
            add_knowledge.dump_embedding_files(texts=files, model_name=model_name, device_type=device, persist_directory=embedding_storing_dir)


if user_choice == "Chat Source Embedding":

    if "embedding_history" not in st.session_state:
        st.session_state.embedding_history = []
    
    st.sidebar.title("Configuration")

    with st.sidebar.expander("Configuration"):
        model_name = st.selectbox(label="Select Your Source Embedding Model: ", options=["hkunlp/instructor-xl"])
        device = st.selectbox(label="Select Your Device: ", options=["cuda", "cpu"])
        embedding_storing_dir = st.selectbox(label="Select Your Database: ", options=list_folder_name(curreny_path= os.getcwd()))
        search_args = st.number_input(label="Number of Searches: ", min_value=1, value=3)
        st.text(" ")
        st.text(" ")

    def generate_answer():
        user_message = st.session_state.input_text
        bot_replys =  ChatSourceEmbedding().embedding_chat(model_name, device, embedding_storing_dir, user_message, search_args)

        st.session_state.embedding_history.append({"message": user_message, "is_user": True})

        for bot_reply in bot_replys:
            st.session_state.embedding_history.append({"message": bot_reply.page_content, "is_user": False})
    

    st.text_input("Talk to the bot", key="input_text")

    if st.button("Send"):
        generate_answer()

    for i, chat in enumerate(reversed(st.session_state.embedding_history)):
        st_message(**chat, key=str(i)) #unpacking



if user_choice == "Wandering Brain":
    
    st.sidebar.title("Configuration")
    with st.sidebar.expander("Configuration"):
        model_name = st.selectbox(label="Select Your Source Embedding Model: ", options=["ggml-gpt4all-j-v1.3-groovy.bin"])

    if "wandering_brain" not in st.session_state:
        st.session_state.wandering_brain = []

    def generate_answer():
        user_message = st.session_state.input_text
        bot_reply =  WanderingBrain().run_gpt4all(model_name, prompt=user_message, model_path=get_model_path(os.getcwd()))

        st.session_state.wandering_brain.append({"message": user_message, "is_user": True})

        
        st.session_state.wandering_brain.append({"message": bot_reply, "is_user": False})
    

    st.text_input("Talk to the bot", key="input_text")

    if st.button("Send"):
        with st.spinner("Thinking.."):
            start_time = time.time()
            generate_answer()
            end_time = time.time()
            execution_time_minutes = (end_time - start_time) / 60

    for i, chat in enumerate(reversed(st.session_state.wandering_brain)):
        st_message(**chat, key=str(i)) #unpacking
    st_message("Execution Time: {:.2f} minutes".format(execution_time_minutes))