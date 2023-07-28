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

def add_pdf_to_DB(pdf_path):
    vectorstore = Chroma(persist_directory=persist_directory, 
                  embedding_function=embeddings)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)
    print('the texts', texts)
    vectorstore.add_documents(texts)

def save_uploadpdf(uploadfile):
        if is_file_processed(uploadfile.filename):
            return (None, False)
        with open(os.path.join("data_pdf", uploadfile.filename), 'wb') as f:
            f.write(uploadfile.file.read())
        return (os.path.join("data_pdf", uploadfile.filename), True)

def choose_search_mode(mode):
    #---------------------
    sys_msg2 = None
    #---------------------
    if mode == "Database Search":
        new_prompt = agent.agent.create_prompt(
            system_message=sys_msg,
            tools=tools
        )
        agent.agent.llm_chain.prompt = new_prompt

        instruction = B_INST + " Respond to the following in JSON with 'action' and 'action_input' values " + E_INST
        human_msg = instruction + "\nUser: {input}"

        agent.agent.llm_chain.prompt.messages[2].prompt.template = human_msg
        print('check template: ', agent.agent.llm_chain.prompt)
    
    elif mode == "Normal Search":
        new_prompt = agent.agent.create_prompt(
            system_message=sys_msg2,
            tools=tools
        )
        agent.agent.llm_chain.prompt = new_prompt

        instruction = B_INST + " Respond to the following in JSON with 'action' and 'action_input' values " + E_INST
        human_msg = instruction + "\nUser: {input}"

        agent.agent.llm_chain.prompt.messages[2].prompt.template = human_msg
        print('check template: ', agent.agent.llm_chain.prompt)
        
        


QA = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())


memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="output"
)

#------------------------------------------------
from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish

class OutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            # this will work IF the text is a valid JSON with action and action_input
            response = parse_json_markdown(text)
            action, action_input = response["action"], response["action_input"]
            if action == "Final Answer":
                # this means the agent is finished so we call AgentFinish
                return AgentFinish({"output": action_input}, text)
            else:
                # otherwise the agent wants to use an action, so we call AgentAction
                return AgentAction(action, action_input, text)
        except Exception:
            # sometimes the agent will return a string that is not a valid JSON
            # often this happens when the agent is finished
            # so we just return the text as the output
            return AgentFinish({"output": text}, text)

    @property
    def _type(self) -> str:
        return "conversational_chat"

# initialize output parser for agent
parser = OutputParser()
#-----------------------------------------------


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
    max_iterations=4,
    memory=memory,
    agent_kwargs={"output_parser": parser}
)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<>\n", "\n<>\n\n"

sys_msg = B_SYS + """AI is a expert JSON builder designed to assist with a wide range of tasks.

AI is able to respond to the User and use tools using JSON strings that contain "action" and "action_input" parameters.

All of AI's communication is performed using this JSON format.

AI can also use tools by responding to the user with tool use instructions in the same "action" and "action_input" JSON format. the Only tools available to AI are:
- "Retrieval Question Answering tool": Use this tool only when the question explicitly states "according to database"; otherwise. Use all contexts from database to answer question.
  - To use the Retrieval Question Answering tool, AI should write like so:
    ```json
    {{"action": "Retrieval Question Answering tool",
      "action_input": give me a summary of document }}
    ```
When you have  multiple contexts and table aggregate all of them to response to questions.

Here are some previous conversations between the AI and User:

User: 1.0 how are you?
AI: ```json
{{"action": "Final Answer",
 "action_input": "I'm good thanks, how are you?"}}
```

User4: where is the capital of Iran?
AI: ```json
{{"action": "Final Answer",
 "action_input": "The capital of Iran is Tehran"}}
```
User: 2.0
AI: ```json
{{"action": "Final Answer",
 "action_input": "According tho the document world war 1 started in 1941"}}
```
User: 16.0
Context:  The document is a research paper.
Context:  The research paper explores the impact of global waming on mental health.
Context:  The paper was written by Dr. Smith and published in a Nature journal.
AI: ```json
{{"action": "Final Answer",
 "action_input": "According to above context document is about theglobal warming and its consequesnces"}}
```
User: 16.0
Context:  The document is a annual report.
Context:  The research paper explores the impact of global waming on mental health.
Context:  The paper was written by Dr. Smith and published in a Nature journal.
AI: ```json
{{"action": "Final Answer",
 "action_input": "According to above context document is about theglobal warming and its consequesnces"}}
```
Here is the latest conversation between AI and User.""" + E_SYS
new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)
agent.agent.llm_chain.prompt = new_prompt

instruction = B_INST + " You are a helpful AI. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'. Respond to the following in JSON with 'action' and 'action_input' values " + E_INST
human_msg = instruction + "\nUser: {input}"

agent.agent.llm_chain.prompt.messages[2].prompt.template = human_msg


# -----------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def inference(prompt):
    print('check template: ', agent.agent.llm_chain.prompt)
    return agent(prompt)

app = FastAPI()

class Input(BaseModel):
      prompt:str
      messages: List[Dict[str, str]]

class SearchModeInput(BaseModel):
    search_mode: str

@app.get("/")
def test_root():
     return {"backend","backend for Falcon"}

@app.post("/clearMem")
def clearMemory():
    agent.memory.clear()

@app.post("/search_mode")
def search_modes(input: SearchModeInput):
    choose_search_mode(input.search_mode)

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
    output = inference(prompt_input.prompt)
    return output


if __name__=="__backend__":
     uvicorn.run(app, port=8002)