import os
from langchain.vectorstores import Chroma
from dotenv import dotenv_values
from langchain.document_loaders import YoutubeLoader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.tools import BaseTool
from math import pi
from typing import Union
from langchain.llms import HuggingFacePipeline
from torch import cuda, bfloat16
import transformers
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.embeddings import HuggingFaceInstructEmbeddings
from typing import List, Dict
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import json
from fastapi import FastAPI, File, UploadFile
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
import langchain
from langchain.llms import HuggingFacePipeline
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate, LLMChain
import json
import textwrap
import torch
import re
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

env_variables = dotenv_values('env.env')
os.environ['OPENAI_API_KEY'] = env_variables['Open_API']


import os




B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

instruction = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"
system_prompt = "You are a helpful assistant, you always only answer for the assistant then you stop. read the chat history to get context"

prompt_template = \
"""The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{chat_history}
Human: {input}
AI:"""


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")



def generate(text):
    prompt = get_prompt(text)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs,
                                 max_new_tokens=512,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id,
                                 )
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, '</s>')
        final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs#, outputs

def parse_text(text):
    pattern = r"\s*Assistant:\s*"
    cleaned_text = re.sub(pattern, "", text)
    wrapped_text = textwrap.fill(cleaned_text, width=100)
    return wrapped_text + '\n\n'

def add_video_to_DB(url):
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        result = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)
        texts = text_splitter.split_documents(result)
        vectorstore_video.add_documents(texts)


def load_processed_files():
    if os.path.exists("processed_files.json"):
        with open("processed_files.json", "r") as f:
            return json.load(f)
    else:
        return []

def save_processed_file(filename):
    processed_files = load_processed_files()
    processed_files.append(filename)
    with open("processed_files.json", "w") as f:
        json.dump(processed_files, f)

def is_file_processed(filename):
    processed_files = load_processed_files()
    return filename in processed_files



def save_uploadpdf(uploadfile):
        if is_file_processed(uploadfile.filename):
            return (None, False)
        with open(os.path.join("data_pdf", uploadfile.filename), 'wb') as f:
            f.write(uploadfile.file.read())
        return (os.path.join("data_pdf", uploadfile.filename), True)


def save_video(video_url):
    if os.path.exists("processed_videos.json"):
        with open("processed_videos.json", "r") as f:
            video_list = json.load(f)
            if video_url not in video_list:
                add_video_to_DB(video_url)
                video_list.append(video_url)
                with open("processed_videos.json", "w") as f:
                    json.dump(video_list, f)
                return True
            else:
                return False
    else:
        # If the file doesn't exist, create it and add the first video URL
        video_list = [video_url]
        with open("processed_videos.json", "w") as f:
            json.dump(video_list, f)
            add_video_to_DB(video_url)
        return True
    



model_id = 'meta-llama/Llama-2-70b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
model_config = transformers.AutoConfig.from_pretrained(
    model_id
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto'
)
model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)


generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    #stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.01,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

llm = HuggingFacePipeline(pipeline=generate_text)
#embeddings = OpenAIEmbeddings()

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="output"
)



Doc_persist_directory = "./Document_db"
video_persist_directory = "./YouTube_db"
# Check if DB1 directory exists, and if not, create it
if not os.path.exists(Doc_persist_directory):
    os.makedirs(Doc_persist_directory)
    print(f"Directory '{Doc_persist_directory}' created successfully.")

# Check if DB2 directory exists, and if not, create it
if not os.path.exists(video_persist_directory):
    os.makedirs(video_persist_directory)
    print(f"Directory '{video_persist_directory}' created successfully.")



embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                      model_kwargs={"device": "cuda"})
vectorstore_doc = Chroma("PDF_store", embeddings, persist_directory=Doc_persist_directory)
vectorstore_doc.persist()



vectorstore_video = Chroma("YouTube_store", embeddings, persist_directory=video_persist_directory)
vectorstore_video.persist()


def add_pdf_to_DB(pdf_path):
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)
    texts = text_splitter.split_documents(pages)
    print('the texts', texts)
    vectorstore_doc.add_documents(texts)

QA_video = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore_video.as_retriever(),memory = memory,output_key= "output")
QA_document = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore_doc.as_retriever(),memory = memory,output_key= "output")






#------------------------------------------------------------------------------------


prompt = PromptTemplate(
    input_variables=['chat_history', "input"],
    template=prompt_template,
)

from langchain.chains import ConversationChain
llm = HuggingFacePipeline(pipeline=generate_text)
chat = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
    prompt=prompt
)



# -----------------------------------------------------------------------------



template = get_prompt(instruction, system_prompt)

prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"], template=template
)

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
    output_key= "output"
)


app = FastAPI()

class Input(BaseModel):
      prompt:str
      messages: List[Dict[str, str]]
      mode: str

class SearchModeInput(BaseModel):
    search_mode: str

class VideoURLs(BaseModel):
    url: str

@app.get("/")
def test_root():
     return {"backend","backend for Falcon"}


@app.post("/clearMem")
def clearMemory():
    memory.clear()


@app.post("/clearDocs")
def clearDatabase():
    vectorstore_doc.delete([])
    with open("processed_files.json", "w") as f:
        json.dump([], f)


@app.post("/document_loading")
def document_loading(file: UploadFile = File(...)):
    file_path, is_new = save_uploadpdf(file)
    if is_new:
        #file_path = save_uploadpdf(file)
        add_pdf_to_DB(file_path)
        save_processed_file(file.filename)
        os.remove(file_path)
        return is_new
    else: 
        return is_new


@app.post("/predict")
def make_prediction(prompt_input:Input):
    msg =None
    if prompt_input.mode == "Document Search": 
        resp =QA_document.run(prompt_input.prompt)
        return {'output':resp}
    elif prompt_input.mode == "Video Search":
        resp = QA_video.run(prompt_input.prompt)
        return {'output':resp}
    else: 
        resp = llm_chain.predict(user_input=prompt_input.prompt)
        resp = parse_text(resp)
        output = {'output':resp}
        return output
    

@app.post("/video_loading")
def add_video_to_db(input:VideoURLs):
    save_video(input.url)

@app.post("/clearvideos")
def clear_video_db():
    vectorstore_video.delete([])
    with open("processed_videos.json", "w") as f:
            json.dump([], f)

if __name__=="__backend__":
     uvicorn.run(app, port=8002)