from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
#TODO Clean these imports
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
from chromaviz import visualize_collection
from langchain.chains.conversation.memory import ConversationSummaryMemory


from typing import List, Dict



#creating the model.
def init_falcon():
    model_name = 'tiiuae/falcon-40b-instruct'

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
            # set quantization configuration to load large Falcon model with less GPU memory
            # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

    model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map='auto'
            )
            # Set model to inference mode
    model.eval()
            #print(f"Model loaded on {device}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    stop_token_ids = [
                tokenizer.convert_tokens_to_ids(x) for x in [
                    ['Human', ':'], ['AI', ':']
                ]
            ]
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
    class StopOnTokens(StoppingCriteria):
                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    for stop_ids in stop_token_ids:
                        if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                            return True
                    return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    F40_pipeline = pipeline(
                    task="text-generation",
                    model = model,
                    tokenizer = tokenizer,
                    return_full_text = True,
                    stopping_criteria=stopping_criteria,
                    max_length= 1024, 
                    temperature= 0.0,
                    repetition_penalty=1.1
            )
    llm = HuggingFacePipeline(pipeline = F40_pipeline)
    return llm
    #embed_model = OpenAIEmbeddings()
llm = init_falcon()

app = FastAPI()

class Input(BaseModel):
      prompt:str
      messages: List[Dict[str, str]]

@app.get("/")
def test_root():
     return {"backend","backend for Falcon"}

@app.post("/predict")
def make_prediction(prompt_input:Input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in prompt_input.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\\n\\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\\n\\n"
    output = llm.predict(f"{string_dialogue} {prompt_input.prompt} Assistant: ")
    return output


if __name__=="__backend__":
     uvicorn.run(app, port=8002)