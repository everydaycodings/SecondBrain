from langchain.embeddings import HuggingFaceInstructEmbeddings
import streamlit as st


@st.cache(allow_output_mutation=True)
def load_embedding_model(model_name, device):

    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name=model_name,
                                                      model_kwargs={"device": device})
    
    return instructor_embeddings