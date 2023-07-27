# Run the following command  streamlit run demo_frontend.py 

from typing import Literal
from attr import dataclass
from langchain import ConversationChain, HuggingFacePipeline
from transformers import AutoTokenizer, pipeline,BitsAndBytesConfig
import transformers
import torch
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.document_loaders import PDFPlumberLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
import os
import streamlit as st
from llama_index import Document, LangchainEmbedding, SimpleDirectoryReader, ListIndex,VectorStoreIndex, StorageContext, GPTVectorStoreIndex, LLMPredictor, ServiceContext, load_index_from_storage
#from llama_index.vector_stores import DeepLakeVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import base64
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from transformers import StoppingCriteria, StoppingCriteriaList
import torch.cuda as cuda
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import YouTubeSearchTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.chains.summarize import load_summarize_chain
import ast
from langchain import SerpAPIWrapper
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationSummaryMemory

import requests
import json
import io

def clear_chat_history():
    requests.post(url="http://127.0.0.1:5000/clearMem")
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    
st.title('EscherCloud AI LLM service - Demo ')
#st.image("Eschercloud.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("---")  
st.sidebar.markdown("<br>", unsafe_allow_html=True)


st.sidebar.subheader("Upload a PDF")
uploaded_pdf = st.sidebar.file_uploader("Add PDF",type=['pdf'])

st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("extract document"):
        if uploaded_pdf is not None:
            file_bytes = io.BytesIO(uploaded_pdf.getvalue())
            url = "http://127.0.0.1:5000/document_loading"
            file = {"file": uploaded_pdf}
            response = requests.post(url, files=file)
            st.sidebar.success('Document parsed.')


st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("---")  
st.sidebar.markdown("<br>", unsafe_allow_html=True)

show_answer = st.sidebar.radio(options=["Database Search", "Normal Search"], label="Type of search")
if st.sidebar.button("Confirm"):
         print('yes')


for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
if prompt := st.chat_input():#(disabled=not replicate_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
    
if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(url="http://127.0.0.1:5000/predict", json={'prompt': prompt,'messages': st.session_state.messages})#data=json.dumps(prompt))#generate_llama2_response(llm,prompt)
                print('resp', response)
                response=response.json()
                if response:
                
                    placeholder = st.empty()
                    full_response = response['output']
                    
                    placeholder.markdown(full_response)
                
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)