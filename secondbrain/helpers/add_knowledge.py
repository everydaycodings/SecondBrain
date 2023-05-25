from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


import streamlit as st
import tempfile, os, glob


class AddKnowledge:

    def __init__(self) -> None:
        pass

    def extract_content(self, files, chunk_size, chunk_overlap):

        with tempfile.TemporaryDirectory() as temp_dir:

            for uploaded_file in files:
                asset_path = os.path.join(temp_dir, str(uploaded_file.name).split(".")[0] + ".pdf")

                with open(asset_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
            
            loader = DirectoryLoader(temp_dir, glob="./*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            texts = text_splitter.split_documents(documents)
            
            return texts
