from langchain.embeddings import HuggingFaceInstructEmbeddings
import streamlit as st
import os

@st.cache_resource
def load_embedding_model(model_name, device):

    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name=model_name,
                                                      model_kwargs={"device": device})
    
    return instructor_embeddings


def list_folder_name(folder_path):

    items = os.listdir(folder_path)
    folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
    return folders