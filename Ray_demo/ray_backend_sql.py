import ray
from langchain.chains.conversation.memory import ConversationBufferMemory
import pandas as pd
from ray import serve
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from starlette.requests import Request
from langchain.vectorstores import Chroma
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from ray.serve.drivers import DAGDriver
import re
from typing import Optional
from pydantic import BaseModel
import textwrap
from fastapi import FastAPI, HTTPException
from langchain.chains import RetrievalQA
import logging
from backend_utils import (
    add_user,
    add_conversation,
    get_all_data,
    delete_user,
    delete_conversation,
)
from backend_utils import (
    check_user_existence,
    retrieve_conversation,
    retrieve_latest_conversation,
    update_conversation,
)
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain import PromptTemplate, LLMChain
from typing import List


# ------------------- Initialize Ray Cluster --------------------
class Input(BaseModel):

    username: Optional[str]
    prompt: Optional[str]
    newchat: Optional[bool]
    conversation_number: Optional[int]
    AI_assistance: Optional[bool]
    collection_name: Optional[str]
# ------------------------------ LLM Deployment -------------------------------

app = FastAPI()

@serve.deployment(
    ray_actor_options={"num_gpus": 0.45},
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 2,
        "target_num_ongoing_requests_per_replica": 5,
    })
@serve.ingress(app)
class PredictDeployment:
    def __init__(self):
        import os

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

        from typing import List, Dict

        import json
        from langchain import PromptTemplate, LLMChain

        self.access_token = os.getenv("Hugging_ACCESS_TOKEN")
        self.model_id = "meta-llama/Llama-2-70b-chat-hf"
        self.device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.DEFAULT_SYSTEM_PROMPT = """\
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        self.instruction = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"
        self.system_prompt = "You are a helpful assistant, you always only answer for the assistant then you stop. read the chat history to get context"

        self.prompt_template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

        Current conversation:
        {chat_history}
        Human: {input}
        AI:"""
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='app.log',  # specify the file name if you want logging to be stored in a file
                    filemode='a'  # append to the log file if it exists
                    )
        
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True
        model_id = "meta-llama/Llama-2-70b-chat-hf"
        self.device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16,
        )

        # begin initializing HF items, need auth token for these
        model_config = transformers.AutoConfig.from_pretrained(model_id)

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.model.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.generate_text = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=True,  # langchain expects the full text
            task="text-generation",
            # we pass model parameters here too
            # stopping_criteria=stopping_criteria,  # without this model rambles during chat
            temperature=0.01,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # mex number of tokens to generate in the output
            repetition_penalty=1.1,  # without this output begins repeating
        )
        self.llm = HuggingFacePipeline(pipeline=self.generate_text)
        # embeddings = OpenAIEmbeddings()

        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="output"
        )
        self.template = self.get_prompt(self.instruction)
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "user_input"], template=self.template
        )

        self.embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"}
        )
        self.Doc_persist_directory = "./Document_db"
        video_persist_directory = "./YouTube_db"
        # self.vectorstore_video = Chroma("YouTube_store", persist_directory=video_persist_directory, embedding_function=self.embeddings)
        self.vectorstore_doc = Chroma(
            persist_directory=self.Doc_persist_directory,
            embedding_function=self.embeddings,
        )
        self.QA_document = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore_doc.as_retriever(),
            memory=self.memory,
            output_key="output",
        )
        # self.QA_video = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.vectorstore_video.as_retriever(),memory = self.memory,output_key= "output")

    def get_collection_based_retriver(self, collection):
        vectorstore_doc = Chroma(
            str(collection),
            persist_directory=self.Doc_persist_directory,
            embedding_function=self.embeddings,
        )
        
        return vectorstore_doc

    def get_prompt(self, instruction):
        SYSTEM_PROMPT = self.B_SYS + self.DEFAULT_SYSTEM_PROMPT + self.E_SYS
        prompt_template = self.B_INST + SYSTEM_PROMPT + instruction + self.E_INST
        return prompt_template

    def cut_off_text(self, text, prompt):
        cutoff_phrase = prompt
        index = text.find(cutoff_phrase)
        if index != -1:
            return text[:index]
        else:
            return text

    def remove_substring(self, string, substring):
        return string.replace(substring, "")

    def cleaning_memory(self):
        print(self.memory.chat_memory.messages)
        self.memory.clear()
        print("Chat History Deleted")

    def generate(self, text):
        prompt = self.get_prompt(text)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            final_outputs = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
            final_outputs = self.cut_off_text(final_outputs, "</s>")
            final_outputs = self.remove_substring(final_outputs, prompt)

        return final_outputs  # , outputs

    def parse_text(self, text):
        pattern = r"\s*Assistant:\s*"
        pattern2 = r"\s*AI:\s*"
        cleaned_text = re.sub(pattern, "", text)
        cleaned_text = re.sub(pattern2, "", cleaned_text)
        wrapped_text = textwrap.fill(cleaned_text, width=100)
        return wrapped_text


    def AI_assistance(
        self,request: Input):
        
            try:
                self.logger.info("Received request: %s", request.dict())
                input_prompt = request.prompt
                AI_assistance = request.AI_assistance
                username = request.username
                new_chat = request.newchat
                conversation_number = request.conversation_number
                collection_name = request.collection_name
                # Initialize memory based on whether it's a new chat or not
                if new_chat:
                    memory = ConversationBufferMemory(
                        memory_key="chat_history", return_messages=True, output_key="output"
                    )
                else:
                    # Retrieve the previous conversation from the database
                    if conversation_number == 0:
                        latest_chat = retrieve_latest_conversation(username)
                        chat_history = latest_chat["content"]
                        conversation_number = latest_chat["conversation_number"]
                        print(
                            f" the latest conversation for {username} with the conversation_number of {conversation_number} retrieved from database"
                        )
                    else:
                        chat_history = retrieve_conversation(username, conversation_number)[
                            "content"
                        ]
                        print(
                            f"chat histroy for {username} with the conversation_number of {conversation_number} retrieved from database"
                        )

                    # Initialize memory from the retrieved conversation
                    retrieve_from_db = json.loads(chat_history)
                    retrieved_messages = messages_from_dict(retrieve_from_db)
                    retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
                    memory = ConversationBufferMemory(
                        chat_memory=retrieved_chat_history, memory_key="chat_history"
                    )

                # Create an LLM chain with the appropriate mode

                if AI_assistance:
                    llm_chain = LLMChain(
                        llm=self.llm,
                        prompt=self.prompt,
                        verbose=False,
                        memory=memory,
                        output_key="output",
                    )
                else:
                    print("Document search")
                    retriever = self.get_collection_based_retriver(collection_name)
                    llm_chain = RetrievalQA.from_chain_type(
                        llm=self.llm,
                        chain_type="stuff",
                        retriever=retriever.as_retriever(),
                        memory=memory,
                        output_key="output",
                    )
                

                # Generate a response based on the mode
                if AI_assistance:
                    response = llm_chain.predict(user_input=input_prompt)
                else:
                    response = llm_chain.run(input_prompt)

                # Store the conversation
                extracted_messages = llm_chain.memory.chat_memory.messages
                ingest_to_db = messages_to_dict(extracted_messages)
                prompt_token_number = len(self.tokenizer.tokenize(input_prompt))
                gen_token_number = len(self.tokenizer.tokenize(response))
                update_conversation(username, conversation_number, ingest_to_db,prompt_token_number,gen_token_number)
                response = self.parse_text(response)
                self.logger.info("Successfully processed the request")
                return {"output": response}
            except ConnectionError as ce:
                self.logger.error("Error processing the request: %s", str(ce))
                # Handle connection errors (for example, interacting with the database or calling APIs)
                return {"output": "An error occurred while processing the request"}
                
            except KeyError as ke:
                # Handle key errors (for example, accessing a key in a dictionary that doesn’t exist)
                self.logger.error("Error processing the request: %s", str(ke))
                return {"output": "An error occurred while processing the request"}
                
            except Exception as e:
                # General exception to catch any other unforeseen errors
                self.logger.error("Error processing the request: %s", str(e))
                return {"output": "An error occurred while processing the request"}
                

            
                
    
    @serve.batch(max_batch_size=2, batch_wait_timeout_s=0.1)
    async def handle_batch(self, requests: List) -> List[str]:
        results = []
        try:
            for request in requests:
                results.append(self.AI_assistance(request)["output"])
        except Exception as e:
            print(f"An error occurred while handling batch: {str(e)}")
            # Optionally, log the error
        return results
    
    @app.post("/inference")
    async def root(self, request: Input):
        try:
            self.logger.info("Received a request to /inference endpoint")
            response = await self.handle_batch(request)
            self.logger.info("Processed the request successfully")
            return response
        except Exception as e:
            self.logger.error("Error in /inference endpoint: %s", str(e))
            return {"output": "An error occurred while processing the request"}


app = PredictDeployment.bind()
serve.api.run(app,route_prefix="/")