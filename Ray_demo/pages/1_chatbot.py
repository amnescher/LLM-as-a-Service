import streamlit as st
import requests
import json
import io
from langchain.document_loaders import YoutubeLoader
import os

port =5001
pdf_directory = "./PDF_dir"
if not os.path.exists(pdf_directory):
            os.makedirs(pdf_directory)
            print(f"Directory '{pdf_directory}' created successfully.")
def clear_chat_history():
    PORT_NUMBER = 8000
    search_choice = "Cleaning memory"
    URL = f"http://localhost:{PORT_NUMBER}/predict/?text=&mode={search_choice}&messages={st.session_state.messages}"
    requests.post(URL)
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def clear_videos():
    PORT_NUMBER = 8000
    Video_URL = ""
    data_type = "Video"
    mode= "clear"
    POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&data_path={Video_URL}&mode={mode}"
    requests.post(POST_URL)
    st.sidebar.success = "Cleared database."
def clear_document():
    PORT_NUMBER = 8000
    Video_URL = ""
    data_type = "Document"
    mode= "clear"
    POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&data_path={Video_URL}&mode={mode}"
    requests.post(POST_URL)
    st.sidebar.success = "Cleared database."

def get_collections():
    PORT = 8000
    data_type = 'Collection'
    mode='get_all'
    POST_URL = f"http://localhost:{PORT}/VectoreDataBase/?data_type={data_type}&mode={mode}"
    response = requests.post(POST_URL)

    collections_data = response.json()
        #selected_collections = st.multiselect("Select Collections:", response)
    collections_list = collections_data.get("collections", [])
    return collections_list

st.title('EscherCloud AI LLM service - Demo ')
#st.image("Eschercloud.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]



st.sidebar.button('Clear Documents', on_click=clear_document)
st.sidebar.button("Clear Chat History", on_click=clear_chat_history)
#st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("---")  
st.sidebar.markdown("<br>", unsafe_allow_html=True)

search_choice = st.sidebar.radio(options=["AI Assistance","Document Search"], label="Type of search")
if search_choice == 'Document Search':
    collection_list = get_collections()
    selected_collection = st.sidebar.selectbox("select collection", collection_list)
    if st.sidebar.button("Confirm"):
        if selected_collection is not None:
                    PORT_NUMBER = 8000
                    mode = "Collection choice"
                    POST_URL = f"http://localhost:{PORT_NUMBER}/predict/?text={selected_collection}&mode={mode}"
                    response = requests.post(POST_URL)
                    print('response collection:', response, response.content)
                    st.sidebar.success(f"Collection {selected_collection} selected!")

for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
if prompt := st.chat_input():#(disabled=not replicate_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
    
if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                PORT_NUMBER = 8000
                URL = f"http://localhost:{PORT_NUMBER}/predict/?text={prompt}&mode={search_choice}&messages={st.session_state.messages}"
                response = requests.post(URL)
                response = response.content.decode()
                if response:
                    placeholder = st.empty()
                    full_response = response
                    placeholder.markdown(full_response)
                
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)