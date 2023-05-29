from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st


@st.cache_resource
def load_gpt4all_model(model_name, model_path, max_token, temp, top_p, top_k):
    local_path = '{}/{}'.format(model_path, model_name)  # replace with your desired local file path

    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]
    # Verbose is required to pass to the callback manager
    model = GPT4All(model=local_path, callbacks=callbacks, verbose=True, n_predict=max_token, temp=temp, top_p=top_p, top_k=top_k)
    template = """The following is a friendly conversation between a human and an AI. The AI is very straightforward with its answer and provides specific details from its context only if asked or required. If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=template
    )
    conversation = ConversationChain(
    llm=model, 
    verbose=True, 
    memory=ConversationBufferWindowMemory(k=2),
    callbacks=callbacks,
    prompt=PROMPT
    )

    return conversation


@st.cache_resource
def load_Llama_cpp_model(model_name, model_path, max_token, temp, top_p, top_k, gpu_layers, gpu_acceleration):
    local_path = '{}/{}'.format(model_path, model_name)  # replace with your desired local file path

    # Callbacks support token-wise streaming
    callbacks = CallbackManager([StreamingStdOutCallbackHandler()])
    # Verbose is required to pass to the callback manager
    
    if gpu_acceleration == False:
        model = LlamaCpp(model_path=local_path, callback_manager=callbacks, verbose=True,cmax_tokens=max_token,temperature=temp,top_p=top_p,top_k=top_k)
    
    else:
        model = LlamaCpp(model_path=local_path, callback_manager=callbacks, verbose=True,cmax_tokens=max_token,temperature=temp,top_p=top_p,top_k=top_k, n_gpu_layers=gpu_layers)

    template = """The following is a friendly conversation between a human and an AI. The AI is very straightforward with its answer and provides specific details from its context only if asked or required. If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=template
    )
    conversation = ConversationChain(
    llm=model, 
    verbose=True, 
    memory=ConversationBufferWindowMemory(k=2),
    callbacks=callbacks,
    prompt=PROMPT
    )

    return conversation


class WanderingBrain:

    def __init__(self) -> None:
        pass


    def run_gpt4all(self, model_name, prompt, model_path, max_token, temp, top_p, top_k):

        try:
            conversation = load_gpt4all_model(model_name, model_path[0], max_token, temp, top_p, top_k)
        except:
            conversation = load_gpt4all_model(model_name, model_path[1], max_token, temp, top_p, top_k)
        
        return conversation.predict(input=prompt)
    
    def run_Llama_cpp(self, model_name, prompt, model_path, max_token, temp, top_p, top_k, gpu_layers, gpu_acceleration=False):

        try:
            conversation = load_Llama_cpp_model(model_name, model_path[0], max_token, temp, top_p, top_k)
        except:
            conversation = load_Llama_cpp_model(model_name, model_path[1], max_token, temp, top_p, top_k)
        
        return conversation.predict(input=prompt)

