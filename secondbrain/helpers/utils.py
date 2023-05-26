from langchain.embeddings import HuggingFaceInstructEmbeddings
import streamlit as st
import os

@st.cache_resource
def load_embedding_model(model_name, device):

    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name=model_name,
                                                      model_kwargs={"device": device})
    
    return instructor_embeddings


def list_folder_name(folder_path):

    test_folder = "do-not-remove"

    try:
        folder_path = "{}/secondbrain/database".format(folder_path)

        items = os.listdir(folder_path)
        folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]

        if test_folder in folders:
            folders.remove(test_folder)
            return folders
        
        else:
            folder_path = "{}/SecondBrain/secondbrain/database".format(folder_path)
            folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
            folders.remove(test_folder)
            return folders
    
    except:
        folder_path = "{}/SecondBrain/secondbrain/database".format(folder_path)
        folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
        folders.remove(test_folder)
        return folders