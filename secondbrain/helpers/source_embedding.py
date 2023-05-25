from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
import tempfile, os


def load_pdf(pdf_file):

    if pdf_file != None:

        with tempfile.TemporaryDirectory() as temp_dir:

            file_path = file_path = os.path.join(temp_dir, pdf_file.name)
            with open(file_path, "wb") as f:
                f.write(pdf_file.getbuffer())

            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()

            return pages


def load_embedding_model(model_name, device):

    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name=model_name,
                                                      model_kwargs={"device": device})
    
    return instructor_embeddings, True


def retriver(embeddings, pages, num):
    db_instructEmbedd = Chroma.from_documents(pages, embeddings, persist_directory="s")
    retriever_sec = db_instructEmbedd.as_retriever(search_kwargs={"k": num})

    return retriever_sec


def source_docs(retriever, prompt):

    docs_result = retriever.get_relevant_documents(prompt)
    return docs_result