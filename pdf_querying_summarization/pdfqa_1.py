from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
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

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

def get_llm(llm_name, model_temperature, max_tokens=256):
    #os.environ['OPENAI_API_KEY'] = api_key
    if llm_name == "Falcon-7B":
        model = "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2"
        tokenizer = AutoTokenizer.from_pretrained(model)
        F7B_pipeline = pipeline(
                task="text-generation",
                model = model,
                tokenizer = tokenizer,
                trust_remote_code = True,
                return_full_text = True,
                max_new_tokens=max_tokens,
                device_map= "auto", 
                max_length= 512, 
                temperature= model_temperature,
                torch_dtype=torch.bfloat16,
                #repetition does 
            )
        llm = HuggingFacePipeline(pipeline = F7B_pipeline)
        embed_model = OpenAIEmbeddings()#LangchainEmbedding(HuggingFaceEmbeddings(model_name='text-embedding-ada-002'))#'sentence-transformers/all-mpnet-base-v2'))
        return llm, embed_model
    else:
        print = "No model with that name yet"
        return print

def save_uploadpdf(uploadfile):
    with open(os.path.join("data_pdf", uploadfile.name), 'wb') as f:
        f.write(uploadfile.getbuffer())
    return st.success("Save File:{} to directory".format(uploadfile.name))

def display(file):
    with open(file, 'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

        st.markdown(pdf_display, unsafe_allow_html=True)

def search_pdf(query,llm_a,embed):
    documents = SimpleDirectoryReader('data').load_data()
    if(llm_a == 'Falcon-7B' and embed =='SBERT'):    
        llm_f,embed_model = get_llm('Falcon-7B', 0.1, max_tokens=1024)
    #embed_model = LangchainEmbedding(OpenAIEmbeddings())
        embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')) #LangchainEmbedding(OpenAIEmbeddings())#
        service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm=llm_f),embed_model=embed_model,
                                                chunk_size=1024)
    #storage_context = StorageContext.from_defaults(vector_store=DeepLakeVectorStore('./data_DL'))

        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        query_engine = index.as_query_engine()
        result = query_engine.query(query)
        print('response', result.response)
        return result.response
    else:
        embed_model = LangchainEmbedding(OpenAIEmbeddings())
        service_context = ServiceContext.from_defaults(embed_model=embed_model,
                                                chunk_size=1024)
    #storage_context = StorageContext.from_defaults(vector_store=DeepLakeVectorStore('./data_DL'))

        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        query_engine = index.as_query_engine()
        result = query_engine.query(query)
        print('response', result.response)
        return result.response


#TODO delete this if not used
def extract_pdf(documents, llm_name, model_temperature):
    llm,embed_model = get_llm(llm_name, model_temperature, max_tokens=1024)



    service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm=llm),embed_model=embed_model,
                                                   chunk_size=1024)
    #storage_context = StorageContext.from_defaults(vector_store=DeepLakeVectorStore('./data_DL'))

    temp_index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context)
    query_engine = temp_index.as_query_engine()#response_mode="tree_summarize")
    
    return query_engine





st.set_page_config(layout='wide')
DEFAULT_TERM_STR = (
    "Make a list of terms and definitions that are defined in the context, "
    "with one pair on each line. "
    "If a term is missing it's definition, use your best judgment. "
    "Write each line as as follows:\nTerm: <term> Definition: <definition>"
)
st.title("PDF Question Answer")

setup_tab, searh_tab = st.tabs(["Setup", "Search"])

#, upload_tab, query_tab , "Upload/Extract", "Query", 

with searh_tab:
    st.title('Semantic search Application')
    st.subheader('Setup')
    llm_name = st.selectbox('Which LLM?', ["Falcon-7B","OpenAI"])
    embed_name = st.selectbox('Which Embedding?', ["SBERT","OpenAI"])
    
    st.subheader("Upload a PDF")
    uploaded_pdf = st.file_uploader('pdf',type=['pdf'])
    if uploaded_pdf is not None:
        col1,col2 = st.columns(2) #([2,1]) 
        with col1:
            input_file = save_uploadpdf(uploaded_pdf)
            pdf_file = "data_pdf/"+uploaded_pdf.name
            pdf_file = display(pdf_file)
        with col2:
            st.success("Search Area")
            query_search = st.text_area("Search your query")
            if st.checkbox("search"):
                st.info("Your query: "+query_search)
                result = search_pdf(query_search, llm_name, embed_name)
                st.text_area("Response: ", result, height=100)
with setup_tab:
    st.subheader("LLM Setup")
    #api_key = st.text_input("Enter your OpenAI API key here", type="password")
    llm_name = st.selectbox('Which LLM?', ["Falcon-7B"])
    model_temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, step=0.1)
    term_extract_str = st.text_area("The query to extract terms and definitions with.", value=DEFAULT_TERM_STR)

#with upload_tab:
 #   st.subheader("Extract and Query Definitions")
  #  document_text = st.file_uploader("Upload a PDF", type=['pdf'])
        
   # if st.button("Extract Terms and Definitions") and document_text:
    #    with st.spinner("Extracting..."):
     #       extracted_terms = extract_pdf([Document(text=document_text)],
       #                                     llm_name,
        #                                    model_temperature)
            #extracted_terms = "document text"  # this is a placeholder!
      #  st.write(extracted_terms)




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