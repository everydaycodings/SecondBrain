from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import streamlit as st


@st.cache_resource
def load_gpt4all_model(model_name, model_path):
    local_path = '{}/{}'.format(model_path, model_name)  # replace with your desired local file path

    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]
    # Verbose is required to pass to the callback manager
    model = GPT4All(model=local_path, callbacks=callbacks, verbose=True)

    conversation = ConversationChain(
    llm=model, 
    verbose=True, 
    memory=ConversationBufferWindowMemory(k=2), callbacks=callbacks
    )

    return conversation

class WanderingBrain:

    def __init__(self) -> None:
        pass


    def run_gpt4all(self, model_name, prompt, model_path):

        try:
            conversation = load_gpt4all_model(model_name, model_path[0])
        except:
            conversation = load_gpt4all_model(model_name, model_path[1])
        
        return conversation.predict(input=prompt)

