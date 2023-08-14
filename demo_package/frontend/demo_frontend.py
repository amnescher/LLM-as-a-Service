import streamlit as st
import requests
import json
import io
from langchain.document_loaders import YoutubeLoader

port =5000

def clear_chat_history():
    requests.post(url=f"http://backend:{port}/clearMem")
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def clear_videos():
    requests.post(url=f"http://backend:{port}/clearvideos")
    st.sidebar.success = "Cleared database."
def clear_document():
    requests.post(url=f"http://backend:{port}/clearDocs")
    st.sidebar.success = "Cleared database."

# Function to upload videos to the endpoint
def upload_videos(user_input, port):
    try:
        YoutubeLoader.from_youtube_url(user_input, add_video_info=True)
        data = {"url": user_input}
        url = f"http://backend:{port}/video_loading"
        response = requests.post(url, json=data)
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
                file_bytes = io.BytesIO(uploaded_pdf.getvalue())
                url = f"http://backend:{port}/document_loading"
                file = {"file": uploaded_pdf}
                response = requests.post(url, files=file)
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
                response = requests.post(url=f"http://backend:{port}/predict", json={'prompt': prompt,'mode':search_choice, 'messages': st.session_state.messages})#data=json.dumps(prompt))#generate_llama2_response(llm,prompt)
                print('resp', response)
                print('Response content:', response.content)
                response=response.json()
                if response:
                
                    placeholder = st.empty()
                    full_response = response['output']
                    
                    placeholder.markdown(full_response)
                
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
