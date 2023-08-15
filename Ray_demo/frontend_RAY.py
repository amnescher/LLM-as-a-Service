# Run the following command  streamlit run demo_frontend.py 

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

# Function to upload videos to the endpoint
def upload_videos(user_input, port):
    try:
        YoutubeLoader.from_youtube_url(user_input, add_video_info=True)
        data = {"url": user_input}
        PORT_NUMBER = 8000
        Video_URL = user_input
        data_type = "Video"
        mode= "add"
        POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&data_path={Video_URL}&mode={mode}"
        response = requests.post(POST_URL)
        st.sidebar.text('Videos Uploaded!' + " ✅")
    except Exception as e:
            st.sidebar.warning("Please enter a valid YouTube video URL.")



st.title('EscherCloud AI LLM service - Demo ')
#st.image("Eschercloud.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]



st.sidebar.button('Clear Videos', on_click=clear_videos)
st.sidebar.button('Clear Documents', on_click=clear_document)
st.sidebar.button("Clear Chat History", on_click=clear_chat_history)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("---")  





st.sidebar.markdown("<br>", unsafe_allow_html=True)


st.sidebar.subheader("Upload a PDF")
uploaded_pdf = st.sidebar.file_uploader("Add PDF",type=['pdf'])
st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("Upload document"):
        if uploaded_pdf is not None: 
                pdf_filename = os.path.join(pdf_directory, uploaded_pdf.name)
                # Save the uploaded PDF to the specified directory
                with open(pdf_filename, "wb") as f:
                    f.write(uploaded_pdf.read())
                PORT_NUMBER = 8000
                mode = "add"
                data_type = "Document"
                POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&data_path={pdf_filename}&mode={mode}"
                response = requests.post(POST_URL)
                st.sidebar.text('Document Uploaded!' + " ✅")


st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("---")  
st.sidebar.markdown("<br>", unsafe_allow_html=True)






st.sidebar.subheader("Upload Videos")
st.sidebar.write("Enter YouTube video URLs below:")
user_input = st.sidebar.text_input("Video URL:")

if st.sidebar.button("Upload Videos"):
    upload_videos(user_input, port)
    

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("---")  
st.sidebar.markdown("<br>", unsafe_allow_html=True)
search_choice = st.sidebar.radio(options=["AI Assistance","Document Search", "Video Search"], label="Type of search")




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