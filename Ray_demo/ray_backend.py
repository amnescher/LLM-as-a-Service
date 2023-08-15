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
import textwrap
# ------------------- Initialize Ray Cluster --------------------

#------------------------------ LLM Deployment -------------------------------

@serve.deployment(ray_actor_options={"num_gpus": 0.5}, autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 10,
    },route_prefix="/predict")
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
        
        

        from langchain import PromptTemplate, LLMChain
        from typing import List, Dict
        
        
        import json
        

        self.access_token = os.getenv('Hugging_ACCESS_TOKEN')
        self.model_id = 'meta-llama/Llama-2-70b-chat-hf'
        self.device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.DEFAULT_SYSTEM_PROMPT = """\
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        self.instruction = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"
        self.system_prompt = "You are a helpful assistant, you always only answer for the assistant then you stop. read the chat history to get context"

        self.prompt_template = \
        """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

        Current conversation:
        {chat_history}
        Human: {input}
        AI:"""

        model_id = 'meta-llama/Llama-2-70b-chat-hf'
        self.device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

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

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto'
        )
        self.model.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.generate_text = transformers.pipeline(
            model=self.model, tokenizer=self.tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            #stopping_criteria=stopping_criteria,  # without this model rambles during chat
            temperature=0.01,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # mex number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
        )
        self.llm = HuggingFacePipeline(pipeline=self.generate_text)
        #embeddings = OpenAIEmbeddings()

        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="output"
        )
        self.template = self.get_prompt(self.instruction)
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "user_input"], template=self.template
        )
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=True,
            memory=self.memory,
            output_key= "output"
        )

        self.embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"})
        Doc_persist_directory = "./Document_db"
        video_persist_directory = "./YouTube_db"
        self.vectorstore_video = Chroma("YouTube_store", persist_directory=video_persist_directory, embedding_function=self.embeddings)
        self.vectorstore_doc = Chroma("PDF_store",persist_directory=Doc_persist_directory, embedding_function=self.embeddings)

        self.QA_video = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.vectorstore_video.as_retriever(),memory = self.memory,output_key= "output")
        self.QA_document = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.vectorstore_doc.as_retriever(),memory = self.memory,output_key= "output")


    def get_prompt(self, instruction):
        SYSTEM_PROMPT = self.B_SYS + self.DEFAULT_SYSTEM_PROMPT +self.E_SYS
        prompt_template =  self.B_INST + SYSTEM_PROMPT + instruction + self.E_INST
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
        with torch.autocast('cuda', dtype=torch.bfloat16):
            inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
            outputs = self.model.generate(**inputs,
                                    max_new_tokens=512,
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    pad_token_id=self.tokenizer.eos_token_id,
                                    )
            final_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            final_outputs = self.cut_off_text(final_outputs, '</s>')
            final_outputs = self.remove_substring(final_outputs, prompt)

        return final_outputs#, outputs
    def parse_text(self,text):
        pattern = r"\s*Assistant:\s*"
        cleaned_text = re.sub(pattern, "", text)
        wrapped_text = textwrap.fill(cleaned_text, width=100)
        return wrapped_text 


    def _run_chain(self, text: str):
        return self.llm_chain(text)

    def _run_docSearch(self, text: str):
        return self.QA_document.run(text)

    def _run_vidSearch(self, text: str):
        return self.QA_video.run(text)


    async def __call__(self, request: Request):
        text = request.query_params["text"]
        mode = request.query_params["mode"]

        if mode == "Document Search":
            print("Document Search")
            response = self._run_docSearch(text)
            return response #self.parse_text(response)
        elif mode == "Video Search":
            print("video Search")
            response = self._run_vidSearch(text)
            return self.parse_text(response)
        elif mode == "Cleaning memory":
             self.cleaning_memory()
        else:
            response = self._run_chain(text)["output"]
            return self.parse_text(response)

app = PredictDeployment.bind()


