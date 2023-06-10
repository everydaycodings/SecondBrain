from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st
import tempfile, os, glob
from helpers.utils import load_embedding_model
from langchain.document_loaders import WikipediaLoader


class AddKnowledge:

    def __init__(self) -> None:
        pass

    def extract_pdf_content(self, files, chunk_size, chunk_overlap):

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
    
    
    def extract_wikepedia_content(self, prompt, chunk_size, chunk_overlap):

        loader = WikipediaLoader(query=prompt, load_max_docs=2)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
            
        return texts

    

    
    def dump_embedding_files(self, texts, model_name, device_type, persist_directory):

        embedding = load_embedding_model(model_name, device_type)

        vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory="SecondBrain/secondbrain/database/{}".format(persist_directory))

        vectordb.persist()
        vectordb = None
        st.success("Knowledge Added To DataBase")
