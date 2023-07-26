# Run the following command uvicorn demo_backend:app --reload --port 5000
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
from langchain.chains.conversation.memory import ConversationSummaryMemory


from typing import List, Dict

# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# Bring in streamlit for UI/app interface
# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma
from dotenv import dotenv_values
env_variables = dotenv_values('env.env')
#os.environ['OPENAI_API_KEY'] = env_variables['OPENAI_API_KEY']
from langchain.document_loaders import YoutubeLoader
from langchain.chat_models import ChatOpenAI
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool
from math import pi
from typing import Union
from langchain.agents import load_tools
from langchain.tools import BaseTool
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

import requests
from PIL import Image
from langchain.tools import BaseTool
from langchain.chains import RetrievalQA

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

class CircumferenceTool(BaseTool):
    name = "Circumference calculator"
    description = "use this tool when you need to calculate a circumference using the radius of a circle"

    def _run(self, radius: Union[int, float]):
        return float(radius)*2.0*pi
    
    def _arun(self, radius: Union[int, float]):
        raise NotImplementedError("This tool does not support async")




desc = (
    "use this tool when given the URL of an image that you'd like to be "
    "described. It will return a simple caption describing the image."
)

class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = desc

    def _run(self, url: str):
        # download the image and convert to PIL object
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        # preprocess the image
        inputs = processor(image, return_tensors="pt").to(device)
        # generate the caption
        out = model.generate(**inputs, max_new_tokens=20)
        # get the caption
        caption = processor.decode(out[0], skip_special_tokens=True)
        caption = " I know the answer, inside the image " + caption
        return caption
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


desc = (
    "use this tool when given the URL of a pdf document and answer any question related to this document."
)
pdf_url_list = []
class PDFRetrival(BaseTool):
    name = "pdf retrival"
    description = desc
    def _run(self, url:str):
        if url not in pdf_url_list:
            pdf_url_list.append(url)
            add_document_to_db(url)
            return "parsed pdf and saved it do vector database. Now use the PDF Retrieval Question Answering tool to answer the question. Input to the PDF Retrieval Question Answering tool is the question your eceived about the video"
        else:
            return "the document already exist in the database. Now use the PDF Retrieval Question Answering tool to answer the question. Input to the PDF Retrieval Question Answering tool is the question your eceived about the video"

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

desc = (
    "use this tool when given the URL of a youtube video and that you need you need to watch video to answer questions about video"
)
url_list = []
class YouTubeRetrival(BaseTool):
    name = "watch video"
    description = desc

    def _run(self, url: str):
        if url not in url_list:
            url_list.append(url)
            add_video_to_DB(url)
            return "video saved into vectort database. Now use Retrieval Question Answering tool to answer question. Input to Retrieval Question Answering tool is the question you received about the video "
        else:
            return "video already existed in the database. Now use Retrieval Question Answering tool to answer question. Input to Retrieval Question Answering tool is the question you received about the video "        
              
        
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")



hf_model = "Salesforce/blip-image-captioning-large"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

llm = ChatOpenAI(
    #openai_api_key=env_variables['OPENAI_API_KEY'],
    temperature=0,
    model_name='gpt-3.5-turbo'
)


embeddings = OpenAIEmbeddings()
vectorstore = Chroma("YouTube_store", embeddings, persist_directory="./YouTube_db")
documentdb = Chroma("document_store", embeddings, persist_directory="./document_db")


def add_video_to_DB(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    result = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    texts = text_splitter.split_documents(result)
    text =[texts[0]]
    vectorstore.add_documents(text)
    vectorstore.persist()

def add_document_to_db(url):
    pdfloader = PyPDFLoader(url)
    pages = pdfloader.load_and_split()
    sections = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len).split_documents(pages)
    print('check pdf parsed: ', sections)
    documentdb.add_documents(sections)#db5 = Chroma.from_documents(sections, embeddings, persist_directory="./chroma_db")
    documentdb.persist()


processor = BlipProcessor.from_pretrained(hf_model)
model = BlipForConditionalGeneration.from_pretrained(hf_model).to(device)

hf_model = "Salesforce/blip-image-captioning-large"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = BlipProcessor.from_pretrained(hf_model)
model = BlipForConditionalGeneration.from_pretrained(hf_model).to(device)

from langchain.tools import BaseTool, StructuredTool, Tool, tool

doc_parse = PDFRetrival()
image_cap = ImageCaptionTool()
calcu = CircumferenceTool()
QA = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
PDFQA = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=documentdb.as_retriever())
watch_video  =YouTubeRetrival()

tools = [
    Tool.from_function(
        func=image_cap.run,
        name="caption",
        description="use this tool only when given the URL of an image that you'd like to be described. It will return a simple string describing the image. once you recived the string don't use any tool and use it as final answer"
        
    ),
    Tool.from_function(
        func=calcu.run,
        name="calculator",
        description="use this tool only when you need to calculate a circumference using the radius of a circle. it receives a floating point number "
        
    ),
    Tool.from_function(
        func=watch_video.run,
        name="Watch Youtube video tool",
        description= "use this tool Only when given the URL of a youtube video and that you need you need to watch video to answer questions about video"
        
    ),
    Tool.from_function(
        func=doc_parse.run,
        name="Document Parser tool",
        description="use this tool only when given the URL of a document or pdf. It will extract all the embeddings of the document and store it in a database. When you are done using this tool you will most likely use the Retrieval Question Answering tool"
        
    ),
    Tool.from_function(
        func=PDFQA.run,
        name="PDF Retrieval Question Answering tool",
        description="use this tool only when you need to answer questions about a document stored using the Document Parser tool. It is designed to answer questions about a document using the URL provided"
        
    ),
    Tool.from_function(
        func=QA.run,
        name="Retrieval Question Answering tool",
        description= "Use this tool only after Watch Youtube video tool. It is designed to answer questions about the video using the URL provided."
        
    )
]


from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish




from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")
from langchain.agents import AgentType
llm=OpenAI(temperature=0)
agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose= False, memory=memory)

def inference(prompt):
    print(agent_chain(prompt))
    return agent_chain(prompt)


app = FastAPI()

class Input(BaseModel):
      prompt:str
      messages: List[Dict[str, str]]

@app.get("/")
def test_root():
     return {"backend","backend for Falcon"}

@app.post("/clearMem")
def clearMemory():
    agent_chain.memory.clear()

@app.post("/predict")
def make_prediction(prompt_input:Input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in prompt_input.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\\n\\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\\n\\n"
    output = inference(prompt_input.prompt)
    return output


if __name__=="__backend__":
     uvicorn.run(app, port=8002)