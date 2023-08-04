# Run the following command  streamlit run demo_frontend.py 

import streamlit as st

import requests
import json
import io

def clear_chat_history():
    requests.post(url="http://backend:5000/clearMem")
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def clear_database():
    requests.post(url="http://backend:5000/clearDatabase")
    #st.sidebar.success = "Cleared database."
    
st.title('EscherCloud AI LLM service - Demo ')
#st.image("Eschercloud.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
st.sidebar.markdown("---") 
st.sidebar.button('Clear Database', on_click=clear_database)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("---")  





st.sidebar.markdown("<br>", unsafe_allow_html=True)


st.sidebar.subheader("Upload a PDF")
uploaded_pdf = st.sidebar.file_uploader("Add PDF",type=['pdf'])

st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("extract document"):
        if uploaded_pdf is not None:
            #st.sidebar.success('Document parsed.')
            file_bytes = io.BytesIO(uploaded_pdf.getvalue())
            url = "http://backend:5000/document_loading"
            file = {"file": uploaded_pdf}
            response = requests.post(url, files=file)
            st.sidebar.success('Document parsed.')
        


st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("---")  
st.sidebar.markdown("<br>", unsafe_allow_html=True)

search_choice = st.sidebar.radio(options=["Database Search", "Normal Search"], label="Type of search")
if st.sidebar.button("Confirm"):
         url = "http://backend:5000/search_mode"
         data = {"search_mode": search_choice}
         response = requests.post(url, json=data)




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
                if search_choice == "Database Search":
                    prompt = " Using Retrieval Question Answering tool " + prompt
                else:
                    prompt = "Without using Retrieval Question Answering tool or any other tools + " + prompt 
                response = requests.post(url="http://backend:5000/predict", json={'prompt': prompt,'messages': st.session_state.messages})#data=json.dumps(prompt))#generate_llama2_response(llm,prompt)
                print('resp', response)
                print('Response content:', response.content)
                response=response.json()
                if response:
                
                    placeholder = st.empty()
                    full_response = response['output']
                    
                    placeholder.markdown(full_response)
                
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)