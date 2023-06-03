from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
from pathlib import Path
from tqdm import tqdm
import os, requests

@st.cache_resource
def load_embedding_model(model_name, device):

    if model_name == "hkunlp/instructor-xl":
        embeddings = HuggingFaceInstructEmbeddings(model_name=model_name,
                                                        model_kwargs={"device": device})
    
    if model_name == "sentence-transformers/all-MiniLM-L6-v2":
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    return embeddings


def list_folder_name(curreny_path):

    test_folder = "do-not-remove"

    try:
        folder_path = "{}/secondbrain/database".format(curreny_path)

        items = os.listdir(folder_path)
        folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]

        if test_folder in folders:
            folders.remove(test_folder)
            return folders
        
        else:
            folder_path = "{}/SecondBrain/secondbrain/database".format(curreny_path)
            items = os.listdir(folder_path)
            folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
            folders.remove(test_folder)
            return folders
    
    except:
        folder_path = "{}/SecondBrain/secondbrain/database".format(curreny_path)
        items = os.listdir(folder_path)
        folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
        folders.remove(test_folder)
        return folders


def list_files(current_path):
    files = []

    try:
        directory = "{}/secondbrain/models".format(current_path)
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                files.append(filename)
        return files
    
    except:
        directory = "{}/SecondBrain/secondbrain/models".format(current_path)
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                files.append(filename)
        return files


def get_model_path(current_path):

    test_folder = "do-not-remove"


    folder_path1 = "{}/secondbrain/models".format(current_path)

    folder_path2 = "{}/SecondBrain/secondbrain/models".format(current_path)
    
    return [folder_path1] +[folder_path2]



def download_model(model_name, model_link, current_path):

    try:
        model_path= get_model_path(current_path)
        local_path = "{}/{}".format(model_path[0], model_name)

        # Example model. Check https://github.com/nomic-ai/gpt4all for the latest models.
        url = model_link

        # send a GET request to the URL to download the file. Stream since it's large
        response = requests.get(url, stream=True)

        #open the file in binary mode and write the contents of the response to it in chunks
        #This is a large file, so be prepared to wait.
        with open(local_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=81920)):
                if chunk:
                    f.write(chunk)
    
    except:
        model_path = get_model_path(current_path)
        local_path = "{}/{}".format(model_path[1], model_name)

        # Example model. Check https://github.com/nomic-ai/gpt4all for the latest models.
        url = model_link

        # send a GET request to the URL to download the file. Stream since it's large
        response = requests.get(url, stream=True)

        #open the file in binary mode and write the contents of the response to it in chunks
        #This is a large file, so be prepared to wait.
        with open(local_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                if chunk:
                    f.write(chunk)


def remove_model(current_path, models_selected):
    
    try:
        directory = "{}/secondbrain/models".format(current_path)
        for models in models_selected:
            file_path = "{}/{}".format(directory, models)
            if os.path.exists(file_path):
                os.remove(file_path)
    
    except:
        directory = "{}/SecondBrain/secondbrain/models".format(current_path)
        for models in models_selected:
            file_path = "{}/{}".format(directory, models)
            if os.path.exists(file_path):
                os.remove(file_path)