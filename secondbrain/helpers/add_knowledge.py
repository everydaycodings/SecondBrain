from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st
import tempfile, os, glob

persist_directory = 'db'

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

        
    def load_embedding_model(self, model_name, device_type):

        instructor_embeddings = HuggingFaceInstructEmbeddings(model_name=model_name,
                                                    model_kwargs={"device": device_type})

        return instructor_embeddings

    
    def dump_embedding_files(self, texts, model_name, device_type):

        embedding = self.load_embedding_model(model_name, device_type)
        st.info("{} Embedding has been Loaded")

        vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)

        vectordb.persist()
        vectordb = None
        st.success("Knowledge Added To DataBase")
