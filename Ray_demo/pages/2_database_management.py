import json
import streamlit as st
import os
import requests
from langchain.document_loaders import YoutubeLoader

port =5001
pdf_directory = "./PDF_dir"
if not os.path.exists(pdf_directory):
            os.makedirs(pdf_directory)
            print(f"Directory '{pdf_directory}' created successfully.")

def upload_videos(user_input, port, collection, video_name):
    try:
        YoutubeLoader.from_youtube_url(user_input, add_video_info=True)
        data = {"url": user_input}
        PORT_NUMBER = 8000
        Video_URL = user_input
        data_type = "Video"
        collection_ = collection
        vid_name = video_name
        mode= "add"
        POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&data_path={Video_URL}&collection={collection_}&mode={mode}&doc_name={vid_name}"
        response = requests.post(POST_URL)
        print('the response upload video: ', response, response.content)
        st.text('Videos Uploaded!' + " ✅")
    except Exception as e:
            st.warning("Please enter a valid YouTube video URL.")

def upload_webpage(user_input, port, collection, name):
        data = {"url": user_input}
        PORT_NUMBER = 8000
        page_URL = user_input
        data_type = "Webpage"
        collection_ = collection
        page_name = name
        mode= "add"
        POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&data_path={page_URL}&collection={collection_}&mode={mode}&doc_name={page_name}"
        response = requests.post(POST_URL)
        st.text('Page Uploaded!' + " ✅")

def display_documents(selected_collections):
    PORT = 8000
    data_type = 'Document'
    mode='get_all'
    collection = selected_collections
    POST_URL = f"http://localhost:{PORT}/VectoreDataBase/?data_type={data_type}&collection={collection}&mode={mode}"
    response = requests.post(POST_URL)
    response_content = response.content.decode('utf-8')
    intermediate_data = json.loads(response_content)  
    document_list = intermediate_data.get("collection", [])
    return document_list
      
def display_colleciton_in_table():
    PORT = 8000
    data_type = 'Collection'
    mode='get_all'
    POST_URL = f"http://localhost:{PORT}/VectoreDataBase/?data_type={data_type}&mode={mode}"
    response = requests.post(POST_URL)
    collections_data = response.json()
    print('resp', collections_data)
    collections_list = collections_data.get("collections", [])
    print('check display', collections_list)

    #collections_list = ', '.join(collections_list) + ", all_collections"
    return collections_list
     
st.title("Vector Database Management Dashboard")

# Fetch and display collections
st.sidebar.header("Collections")

# Display documents in the main area
st.header("Documents")

col1, col2 = st.columns(2)
PORT = 8000
data_type = 'Collection'
mode='get_all'
POST_URL = f"http://localhost:{PORT}/VectoreDataBase/?data_type={data_type}&mode={mode}"
response = requests.post(POST_URL)

collections_data = response.json()

collections_list = collections_data.get("collections", [])

with col1:
        collection_name = st.text_input("Enter collection name:")
        collection_name = collection_name.replace(" ", "-")
        if st.button("Create Collection"):
                if collection_name is not None:
                    print('col name', collection_name)
                    data_type = "Collection"
                    mode = "add"
                    PORT_NUMBER = 8000
                    POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&data_path={collection_name}&mode={mode}"
                    response = requests.post(POST_URL)
                    st.success(f"Collection {collection_name} added!")
                    documents = display_colleciton_in_table()

        collection_to_delete = st.text_input("Collection to delete:")
        if st.button("Delete Collection"):
            if collection_to_delete is not None:
                PORT_NUMBER = 8000
                mode = "remove"
                data_type = "Collection"
                POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&data_path={collection_to_delete}&mode={mode}"
                response = requests.post(POST_URL)
                st.success(f"Collection {collection_to_delete} deleted!")
                documents = display_colleciton_in_table()
    
with col2:
        #selected_collections = st.multiselect("Select Collections:", collections_list)
        documents = display_colleciton_in_table()  # Fetch actual documents
        st.table(documents)

st.header("Select collections to use")
selected_collections = st.selectbox("Select Collections:", documents)

doc_list = display_documents(selected_collections)

st.header("Collection selected", selected_collections)

col1_2, col2_2 = st.columns(2)

st.markdown("<br>", unsafe_allow_html=True)

with col1_2:
    management_mode = st.radio(options=["Upload Document","Upload Video", "Upload Webpage"], label="Type upload")
    if management_mode == 'Upload Document':      
        st.subheader("Add Document")
        uploaded_pdf = st.file_uploader("Add PDF",type=['pdf'])
        document_name = st.text_input("Enter document name:")
        document_name = document_name.replace(" ", "-")
        document_metadata = st.text_input("Enter document metadata:")
        if st.button("Upload document"):
                if uploaded_pdf is not None: 
                        pdf_filename = os.path.join(pdf_directory, uploaded_pdf.name)
                        # Save the uploaded PDF to the specified directory
                        with open(pdf_filename, "wb") as f:
                            f.write(uploaded_pdf.read())
                        if document_name == "" or document_name == None:
                              document_name = uploaded_pdf.name
                        PORT_NUMBER = 8000
                        mode = "add"
                        data_type = "Document"
                        collection_ = selected_collections
                        doc_name = document_name
                        POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&data_path={pdf_filename}&mode={mode}&collection={collection_}&doc_name={doc_name}"
                        response = requests.post(POST_URL)
                        st.text('Document Uploaded!' + " ✅")
                        doc_list = display_documents(selected_collections)
    elif management_mode == 'Upload Video':
        st.subheader("Upload Videos")
        st.write("Enter YouTube video URLs below:")
        video_name = st.text_input("name of the video")
        video_name = video_name.replace(" ", "-")
        user_input = st.text_input("Video URL:")
        if st.button("Upload Videos"):
            upload_videos(user_input, port, selected_collections, video_name)
    elif management_mode == 'Upload Webpage':
        st.subheader("Upload Webpage")
        st.write("Enter a webpage URLs below:")
        page_name = st.text_input("name of the page")
        page_name = page_name.replace(" ", "-")
        url = st.text_input("page URL:")
        if st.button("Upload Webpage"):
            upload_webpage(url, port, selected_collections, page_name)
    st.subheader("Remove Document")
    selected_document = st.selectbox("Select document:", doc_list)
    print('selected doc', selected_document)
    if st.button('Remove document'):
              if selected_document is not None:
                    PORT_NUMBER = 8000
                    mode = "remove"
                    data_type = "Document"
                    collection_rm = selected_collections
                    doc_rm_name = selected_document
                    POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&mode={mode}&remove={collection_rm}&doc={doc_rm_name}"
                    response = requests.post(POST_URL)
                    st.text('Document Removed!' + " ✅")
with col2_2:
            doc_list = display_documents(selected_collections)
            st.header("Available documents in the collection")
            if doc_list:
                st.table(doc_list)
            else:
                  st.write('No documents to display')