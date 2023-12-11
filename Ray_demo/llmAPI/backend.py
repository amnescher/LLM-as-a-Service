from langchain.chains.conversation.memory import ConversationBufferMemory
from ray import serve
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
import json
import re
from typing import Optional
from pydantic import BaseModel
import textwrap
from fastapi import FastAPI
from langchain.chains import RetrievalQA
import logging

from langchain.vectorstores import Weaviate
import weaviate
import wandb
from backend_database import Database
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain import PromptTemplate, LLMChain
from typing import List
import json
import os
import yaml
import time


# ------------------- Configuration --------------------
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


with open("cluster_conf.yaml", 'r') as file:
    config = yaml.safe_load(file)
    config = Config(**config)

# ------------------- Initialize Ray Cluster --------------------
class Input(BaseModel):
    username: Optional[str]
    prompt: Optional[str]
    memory: Optional[bool]
    conversation_number: Optional[int]
    AI_assistance: Optional[bool]
    collection_name: Optional[str]
    llm_model: Optional[str]


# ------------------------------ LLM Deployment -------------------------------

app = FastAPI()
@serve.deployment()
@serve.ingress(app)
class PredictDeployment:
    def __init__(self, model_id,
                 temperature =  0.01,
                 max_new_tokens=  512,
                 repetition_penalty= 1.1,
                 batch_size= 2):
        import os
        from langchain.llms import HuggingFacePipeline
        from torch import cuda, bfloat16
        import transformers
        from langchain.chains import RetrievalQA
        from langchain import PromptTemplate
        from langchain.vectorstores import Weaviate

        # class initialization
        try:
            self.weaviate_client = weaviate.Client(
                url=config.weaviate_client_url,   
            )
        except:
            self.logger.error("Error in connecting to Weaviate")
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.batch_size = batch_size

        #setting up weight and bias logging
        self.wandb_logging_enabled = config.WANDB_ENABLE
        if self.wandb_logging_enabled:
            try:
                wandb.login(key = config.WANDB_KEY)
                wandb.init(project="Service Metrics", notes="custom step")
                # Define the custom x axis metric
                wandb.define_metric("The number of input tokens")
                wandb.define_metric("The number of generated tokens")
                wandb.define_metric("Inference Time")
                wandb.define_metric("token/second")
            except:
                self.wandb_logging_enabled = False
                pass
        # load config
        with open("cluster_conf.yaml", "r") as self.file:
            self.config = yaml.safe_load(self.file)
            self.config = Config(**self.config)

        self.access_token = config.Hugging_ACCESS_TOKEN
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

        # setting up logging for debugging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="app.log",  # specify the file name if you want logging to be stored in a file
            filemode="a",  # append to the log file if it exists
        )

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True
        
        # set quantization configuration to load large model 
        self.device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
     
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16,
        )

        # begin initializing HF items, need auth token for these
        model_config = transformers.AutoConfig.from_pretrained(self.model_id, use_auth_token = self.access_token)

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
            use_auth_token = self.access_token
        )
        self.model.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, padding_side="left"
        )
        self.generate_text = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=True,  # langchain expects the full text
            task="text-generation",
            # we pass model parameters here too
            # stopping_criteria=stopping_criteria,  # without this model rambles during chat
            temperature=self.temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=self.max_new_tokens,  # mex number of tokens to generate in the output
            repetition_penalty=self.repetition_penalty,  # without this output begins repeating
            batch_size=self.batch_size,  # number of independent sequences to generate
        )
        self.llm = HuggingFacePipeline(pipeline=self.generate_text)
        self.generate_text.tokenizer.pad_token_id = self.tokenizer.eos_token_id
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

        self.weaviate_vectorstore = Weaviate(
            self.weaviate_client, 
            "Admin_General_collection",
            'page_content', 
            attributes=['page_content']
        )
        # self.vectorstore_video = Chroma("YouTube_store", persist_directory=video_persist_directory, embedding_function=self.embeddings)
        self.QA_document = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.weaviate_vectorstore.as_retriever(),
            memory=self.memory,
            output_key="output",
        )

        # self.QA_video = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.vectorstore_video.as_retriever(),memory = self.memory,output_key= "output")
        self.database = Database()

    def get_collection_based_retriver(self, client, collection):
        content = ['page_content']
        weaviate_vectorstore = Weaviate(
            client,
            str(collection),
            'page_content', 
            attributes=['page_content']
        )
        return weaviate_vectorstore
    
    
    '''def get_collection_based_retriver(self, collection):
        vectorstore_doc = Chroma(
            str(collection),
            persist_directory=self.Doc_persist_directory,
            embedding_function=self.embeddings,
        )

        return vectorstore_doc'''

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

    def AI_assistance(self, request: Input):
        try:
            self.logger.info("Received request: %s", request.dict())
            input_prompt = request.prompt
            AI_assistance = request.AI_assistance
            username = request.username
            memory = request.memory
            conversation_number = request.conversation_number
            collection_name = request.collection_name
            # Initialize memory based on whether it's a new chat or not
            if not memory:
                memory = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True, output_key="output"
                )
            else:
                # Retrieve the latest conversation from the database
                if conversation_number <= 0 or conversation_number is None:
                    latest_chat = self.database.retrieve_conversation(
                        {
                            "username": username,
                        }
                    )

                    chat_history = latest_chat["content"]
                    conversation_number = latest_chat["conversation_number"]
                    print(
                        f" the latest conversation for {username} with the conversation_id of {conversation_number} retrieved from database"
                    )
                else:
                    chat_history = self.database.retrieve_conversation(
                        {
                            "username": username,
                            "conversation_number": conversation_number,
                        }
                    )
                    chat_history = chat_history["content"]
                    print(
                        f"chat histroy for {username} with the conversation_number of {conversation_number} retrieved from database"
                    )

                # Initialize memory from the retrieved conversation
                
                retrieved_messages = messages_from_dict(json.loads(chat_history))
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
                #check if collection exists
                #if self.database.check_collection_exists(request.dict()):
                    #if username[0].isalpha():
                    #    new_username= username[0].upper() + username[1:]
                    collection_name = "Test_user_2_document"#f"{new_username}_{collection_name}"
                    retriever = self.get_collection_based_retriver(self.weaviate_client,collection_name)
                    
                    llm_chain = RetrievalQA.from_chain_type(
                        llm=self.llm,
                        chain_type="stuff",
                        retriever=retriever.as_retriever(),
                        memory=memory,
                        output_key="output",
                    )
                #else: 
                 #   return {"output": "Error: Collection does not exist"}
            pre_inference_memo = llm_chain.memory.chat_memory.messages
            pre_inference_memo = " ".join(
                message.content for message in pre_inference_memo
            )
            pre_inference_memo_token_len = len(
                self.tokenizer.tokenize(pre_inference_memo)
            )
            # Generate a response based on the mode
            inference_start_time = time.time()

            # Generate a response based on the mode
            if AI_assistance:
                response = llm_chain.predict(user_input=input_prompt)
            else:
                #v1 = retriever.asimilarity_search_by_vector(input_prompt, top_k=2)
                #self.logger.info("v1 is %s", v1)
                #v2 = retriever.similarity_search_by_vector(input_prompt, top_k=2)
                #self.logger.info("v2 is %s", v2)
                #v3 = retriever.similarity_search_with_relevance_scores(input_prompt, top_k=2)
                #self.logger.info("v3 is %s", v3)
                #v4 = retriever.max_marginal_relevance_search(input_prompt, top_k=2)
                #self.logger.info("v4 is %s", v4)
                response = llm_chain.run(input_prompt)


            # End measuring time after inference
            inference_end_time = time.time()

            # Calculate and log the elapsed time
            inference_elapsed_time = inference_end_time - inference_start_time

            # Store the conversation
            extracted_messages = llm_chain.memory.chat_memory.messages
            ingest_to_db = messages_to_dict(extracted_messages)
            input_token_number = (
                len(self.tokenizer.tokenize(input_prompt))
                + pre_inference_memo_token_len
            )
            gen_token_number = len(self.tokenizer.tokenize(response))
            db_response = self.database.update_conversation(
                {
                    "username": username,
                    "content": json.dumps(ingest_to_db),
                    "gen_token_number": gen_token_number,
                    "prompt_token_number": input_token_number,
                    "conversation_number": conversation_number,
                }
            )
            response = self.parse_text(response)

            wandb_log = {
                "The number of input tokens": input_token_number,
                "The number of generated tokens": gen_token_number,
                "Inference Time": inference_elapsed_time,
                "token/second": gen_token_number / inference_elapsed_time,
            }
            if self.wandb_logging_enabled:   
                wandb.log(wandb_log)
            self.logger.info("Processed the request successfully")
            return {"output": response}

        except ConnectionError as ce:
            self.logger.error("Error processing the request: %s", str(ce))
            # Handle connection errors (for example, interacting with the database or calling APIs)
            return {"output": "ConnectionError: An error occurred while processing the request"}

        except KeyError as ke:
            # Handle key errors (for example, accessing a key in a dictionary that doesn’t exist)
            self.logger.error("Error processing the request: %s", str(ke))
            return {"output": " KeyError: An error occurred while processing the request"}

        except Exception as e:
            # General exception to catch any other unforeseen errors
            self.logger.error("Error processing the request: %s", str(e))
            return {"output": "Exception : An error occurred while processing the request"}

    @serve.batch(
        max_batch_size=config.max_batch_size,
        batch_wait_timeout_s=config.batch_wait_timeout_s,
    )
    async def handle_batch(self, requests: List) -> List[str]:
        results = []
        self.logger.info(
            "Received a batch of request with batch size of: %s ", len(requests)
        )
        try:
            for request in requests:
                results.append(self.AI_assistance(request)["output"])
        except Exception as e:
            print(f"An error occurred while handling batch: {str(e)}")
            # Optionally, log the error

        return results

    @app.post("/")
    async def root(self, request: Input):
        try:
            self.logger.info("Received requests to /inference endpoint")
            response = await self.handle_batch(request)
            return response
        except Exception as e:
            self.logger.error("Error in /inference endpoint: %s", str(e))
            return {"output": "An error occurred while processing the request"}





#serve.run(PredictDeployment.bind(), route_prefix="/")