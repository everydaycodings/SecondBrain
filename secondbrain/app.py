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
from helpers.utils import list_folder_name, get_model_path, download_model, list_files
from helpers.wandering_brain import WanderingBrain

# Set the theme
st.set_page_config(
    page_title="SecondBrain",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("SecondBrain 🧠")
st.markdown("Store your knowledge and query it with your favorite Open Source Model")

st.markdown("---\n\n")

user_choice = st.sidebar.selectbox("Select Your Choice: ", options=['Add Knowledge', "Chat Source Embedding", "Wandering Brain", 'Chat with your Brain', 'Forget', "Explore", "Utils"])


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
        model_architecture = st.selectbox("Select the model Architecture: ", options=["GPT4ALL", "Llama-cpp"])
        model_name = st.selectbox(label="Select Your Source Model: ", options=list_files(os.getcwd()))
        max_token = st.number_input(label="The maximum number of tokens to generate: ", min_value=1, value=256)
        temp = st.slider(label="The temperature to use for sampling.", min_value=0.0, max_value=1.0, value=0.8)
        top_p = st.slider(label="The top-p value to use for sampling.", min_value=0.0, max_value=1.0, value=0.95)
        top_k = st.slider(label="The top-k value to use for sampling.", min_value=1, max_value=100, value=40)


    if "wandering_brain" not in st.session_state:
        st.session_state.wandering_brain = []

    def generate_answer():
        user_message = st.session_state.input_text

        bot_reply =  WanderingBrain().run_model(
            model_name = model_name, prompt=user_message, model_path=get_model_path(os.getcwd()),
            max_token=max_token, temp=temp, top_p=top_p, top_k=top_k, model_architecture=model_architecture )
        
        
        st.session_state.wandering_brain.append({"message": user_message, "is_user": True})

        
        st.session_state.wandering_brain.append({"message": bot_reply, "is_user": False})
    

    st.text_input("Talk to the bot", key="input_text")

    if st.button("Send"):
        with st.spinner("Thinking.."):
            start_time = time.time()
            generate_answer()
            end_time = time.time()
            execution_time_minutes = (end_time - start_time) / 60

    try:
        st_message("Current Chat Execution Time: {:.2f} minutes".format(execution_time_minutes))
    except:
        pass
    for i, chat in enumerate(reversed(st.session_state.wandering_brain)):
        st_message(**chat, key=str(i)) #unpacking



if user_choice == "Utils":

    st.sidebar.title("Options")
    if st.sidebar.checkbox(label="Download Models", value=True):
        st.title("Download Your Model")
        model_name = st.text_input("Enter The Model Name: ", placeholder='model.bin')
        model_link = st.text_area("Enter The Model Link: ", placeholder='http://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin')

        if st.button("Start Downloading"):
            with st.spinner("Downloading..."):
                download_model(model_name=model_name, model_link=model_link, current_path=os.getcwd())
                st.success("Model Download Completed!!")