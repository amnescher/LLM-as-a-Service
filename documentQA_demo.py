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

# env_variables = dotenv_values('env.env')
# os.environ['OPENAI_API_KEY'] = env_variables['Open_API']


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
persist_directory = "./PDF_db"
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                      model_kwargs={"device": "cuda"})
vectorstore = Chroma("PDF_store", embeddings, persist_directory=persist_directory)
vectorstore.persist()



def add_pdf_to_DB(pdf_path):
    vectorstore = Chroma(persist_directory=persist_directory, 
                  embedding_function=embeddings)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    texts = text_splitter.split_documents(pages)
    text =texts[0:3]
    vectorstore.add_documents(text)
    


QA = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())


memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="output"
)




tools = [
    Tool.from_function(
        func=QA.run,
        name="Retrieval Question Answering tool",
        description= "Use this tool only when a document is uploaded and you want to answer questions about document."  
    )
]

# initialize agent
agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    early_stopping_method="generate",
    memory=memory,
    #agent_kwargs={"output_parser": parser}
)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<>\n", "\n<>\n\n"

sys_msg = B_SYS + """Assistant is a expert JSON builder designed to assist with a wide range of tasks.

Assistant is able to respond to the User and use tools using JSON strings that contain "action" and "action_input" parameters.

All of Assistant's communication is performed using this JSON format.

Assistant can also use tools by responding to the user with tool use instructions in the same "action" and "action_input" JSON format. the Only tools available to Assistant are:
- "Retrieval Question Answering tool": Use this tool only when given a document you need to answer questions about the document.
  - To use the Retrieval Question Answering tool, Assistant should write like so:
    ```json
    {{"action": "Retrieval Question Answering tool",
      "action_input": give me a summary of document }}
    ```

Here are some previous conversations between the Assistant and User:

User: 1.0 how are you?
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "I'm good thanks, how are you?"}}
```

User: 3.0  What is the title of the document?
Assistant: ```json
{{"action": "Retrieval Question Answering tool",
 "action_input": "What is the title of the document?" }}
```
User4: where is the capital of Iran?
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "The capital of Iran is Tehran"}}
```
User: 2.0
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "According tho the document world war 1 started in 1941"}}
```
User: Thanks could you tell me the circumference of a circle that has a radius of 4 mm?
Assistant: ```json
{{"action": "circumference",
 "action_input": "4" }}
```
User: 16.0
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "according to docuemnt the inflation between 1931-2000 wa 11%"}}
```
User: 16.0
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "the document is about the global warming"}}
```

Here is the latest conversation between Assistant and User.""" + E_SYS
new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)
agent.agent.llm_chain.prompt = new_prompt

instruction = B_INST + " Respond to the following in JSON with 'action' and 'action_input' values " + E_INST
human_msg = instruction + "\nUser: {input}"

agent.agent.llm_chain.prompt.messages[2].prompt.template = human_msg


# -----------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def inference(prompt):
    return agent(prompt)

app = FastAPI()

class Input(BaseModel):
      prompt:str
      messages: List[Dict[str, str]]

@app.get("/")
def test_root():
     return {"backend","backend for Falcon"}

@app.post("/clearMem")
def clearMemory():
    agent.memory.clear()

@app.post("/document_loading")
def document_loading():
    add_pdf_to_DB("")

@app.post("/predict")
def make_prediction(prompt_input:Input):
    output = inference(prompt_input.prompt)
    return output


if __name__=="__backend__":
     uvicorn.run(app, port=8002)