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
from chromaviz import visualize_collection
from langchain.chains.conversation.memory import ConversationSummaryMemory

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


###############################################################################################
##### Chatbot functions
############################################################################################### 










@st.cache(allow_output_mutation=True)
def get_llm(llm_name, model_temperature, max_tokens=256):
    #os.environ['OPENAI_API_KEY'] = api_key
    #if llm_name == "Falcon-7B":
     #   model = "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2"
      #  tokenizer = AutoTokenizer.from_pretrained(model)
       # F7B_pipeline = pipeline(
        #        task="text-generation",
         #       model = model,
          #      tokenizer = tokenizer,
           #     trust_remote_code = True,
            #    return_full_text = True,
             #   max_new_tokens=max_tokens,
              #  device_map= "auto", 
               # max_length= 512, 
                #temperature= model_temperature,
                #torch_dtype=torch.bfloat16,
                #repetition does 
            #)
        #llm = HuggingFacePipeline(pipeline = F7B_pipeline)
        #embed_model = OpenAIEmbeddings()#LangchainEmbedding(HuggingFaceEmbeddings(model_name='text-embedding-ada-002'))#'sentence-transformers/all-mpnet-base-v2'))
        #return llm, embed_model
    if llm_name == "Falcon-40B":
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
        embed_model = OpenAIEmbeddings()
        return llm,embed_model
    else:
        print = "No model with that name yet"
        return print,print




def load_embeddings(embeddings):
    if embeddings == "OpenAI ADA-002":
            embed_model = OpenAIEmbeddings()
    elif embeddings == "SBERT":
            embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    return embed_model

def save_uploadpdf(uploadfile):
    with open(os.path.join("data_pdf", uploadfile.name), 'wb') as f:
        f.write(uploadfile.getbuffer())
    return st.success("Save File:{} to directory".format(uploadfile.name))

def parse_pdf(file, embeddings):
    print('parsing pdf:', file, 'using embeddings:', embeddings)
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    sections = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap=100, length_function=len).split_documents(pages)
    print('check pdf parsed: ', sections)
    db5 = Chroma.from_documents(sections, embeddings, persist_directory="./chroma_db")
    db5.persist()
    print("finished parsing PDF")

def display(file):
    with open(file, 'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

        st.markdown(pdf_display, unsafe_allow_html=True)

def confirm_search_options(answer,summary,youtube,web_search):
    options = [answer,summary,youtube,web_search]
    print('the options selected:', options)
    return options

def search_pdf(query,llm,embeddings,options): 
        if not options:
            options = [False, False, False, False]
        print('options selected: ', options) 
        db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        if options[0] == True:
            gri_std = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=db3.as_retriever())
            answer = gri_std.run(query)
        else:
            answer = None
        if options[1] == True:
            chain = load_summarize_chain(llm, chain_type="stuff")
            search = db3.similarity_search(query)
            summary = chain.run(input_documents=search, question="Write a summary within 200 words.")
        else: 
            summary = None
        if options[2] == True:
            youtubeSearch = YouTubeSearchTool()
            resource_youtube = youtubeSearch.run(query)
            resource_youtube = ast.literal_eval(resource_youtube)
            resource_youtube = ["https://www.youtube.com" + link for link in resource_youtube]
        else:
            resource_youtube = None
        if options[3] == True:
            search = SerpAPIWrapper(serpapi_api_key="fcd97e685e32bfb8329f25d1c23d4dfd10707223c40417cd24969b8f43a68222")
            search_query = query
            internet_resource = search.run(search_query)
        else:
            internet_resource = None
        print('response', answer, 'youtube=', resource_youtube)

        print('combined summary', summary)
        return answer, resource_youtube, summary, internet_resource

def display_chormadb(embeddings):
    chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    visualize_collection(chroma_db._collection)



st.set_page_config(layout='wide')

st.title("PDF Question Answer")

search_tab, chroma_db_tab, chatbot_tab = st.tabs(["Search","ChromaDB","ChatBot"])

#, upload_tab, query_tab , "Upload/Extract", "Query", 

with search_tab: 
    st.title('Semantic search Application')
    st.subheader('Setup')
    llm_name = st.selectbox('Which LLM?', ["Falcon-7B","Falcon-40B"])
    embedding_name = st.selectbox('Which Embeddings?', ["SBERT","OpenAI ADA-002"])
    model,embedding = get_llm(llm_name, 0.0, 1024)

    #embedding = load_embeddings(embedding_name)
    st.subheader("Upload a PDF")
    uploaded_pdf = st.file_uploader('pdf',type=['pdf'])
   
    if uploaded_pdf is not None:
        
        col1,col2,col3 = st.columns(3) #([2,1]) 
        with col1:
            input_file = save_uploadpdf(uploaded_pdf)  
            pdf_file = "data_pdf/"+uploaded_pdf.name
            if st.button("Extract PDF"):
                parse_pdf(pdf_file,embedding)
            pdf_file = display(pdf_file)
        with col3:
            st.subheader('Select output')
            show_answer = st.checkbox('Show answer')
            show_summary = st.checkbox('Show Summary')
            show_websearch = st.checkbox('Additonal resources')
            show_youtube = st.checkbox('Youtube additional resource')

        with col3:
            st.success("Search Area")
            query_search = st.text_area("Search your query")
            if st.button("Search"):
                st.info("Your query: "+query_search)
                options = confirm_search_options(show_answer,show_summary,show_youtube, show_websearch)
                result, resource, summary,wresource = search_pdf(query_search, model,embedding,options)
                if result is not None:
                    st.text_area("Response: ", result, height=70)
                if summary is not None:
                    st.text_area("Summary of the article: ", summary, height=70)
                if resource is not None:
                    st.text_area("Youtube Resource: ", "\n".join(resource), height=70)
                if wresource is not None:
                    st.text_area("Resource: ", wresource, height=70)

with chroma_db_tab:
    st.title('Semantic search Application')
    st.subheader('Setup')
    if st.button("Display DB"):
    
    
        display_chormadb(embedding)

with chatbot_tab:
    st.title('Falcon Chatbot')
    st.subheader('Setup')



###############################################################################
# END OF STREAMLIT
###############################################################################

#Question that works:
#   What are the GRI standards?
#   "how is the sustainability performance reported?"
#   "what is the impact of the GRI guidelines?"
#   "who wrote this article?"
#   "who uses the GRI standards?"
#   "what is the impact of the GRI guidelines?"
#   "who is impacted by the GRI guidelines?"


def main():
    print('test')
  #  pipeline = create_model()
   # print(pipeline)

main()