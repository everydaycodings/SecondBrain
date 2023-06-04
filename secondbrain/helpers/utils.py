from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
from pathlib import Path
from tqdm import tqdm
import os, requests,zipfile, tempfile, base64
from datetime import datetime

@st.cache_resource
def load_embedding_model(model_name, device):

    if model_name == "hkunlp/instructor-xl":
        embeddings = HuggingFaceInstructEmbeddings(model_name=model_name,
                                                        model_kwargs={"device": device})
    
    if model_name == "sentence-transformers/all-MiniLM-L6-v2":
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    return embeddings



def download_button(encoded, file_name):

    now = datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d_%H_%M_%S")

    st.markdown(
                    f"""
                    <a href="data:application/zip;base64,{encoded}" download="{file_name}_{formatted_date_time}.zip">
                        <h3>Download</h3>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )



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



def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))


def export_database(database_name, current_path):
    folder_path = "{}/SecondBrain/secondbrain/database/{}".format(current_path, database_name)
    with tempfile.TemporaryDirectory() as temp_dir:
        output_zip_path = os.path.join(temp_dir, '{}.zip'.format(database_name))
        zip_folder(folder_path, output_zip_path)

        file_size = os.path.getsize(output_zip_path)

        if file_size == 22:
            folder_path = "{}/secondbrain/database/{}".format(current_path, database_name)
            output_zip_path = os.path.join(temp_dir, '{}.zip'.format(database_name))
            zip_folder(folder_path, output_zip_path)

    
        with open(output_zip_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        
        download_button(encoded=encoded, file_name="{}-export-database".format(database_name))