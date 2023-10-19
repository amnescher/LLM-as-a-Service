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
     
def weaviate_get_classes():
    PORT = 8000
    data_type = 'Weaviate'
    mode='get_all'
    POST_URL = f"http://localhost:{PORT}/VectoreDataBase/?data_type={data_type}&mode={mode}"
    response = requests.post(POST_URL)
    if response.status_code == 200:
        try:
            weaviate_data = response.json()
            weaviate_list = weaviate_data.get("weaviate", [])
            print('weaviate list', weaviate_list)
            return weaviate_list
        except requests.exceptions.JSONDecodeError:
            print(f"Failed to decode JSON. Response content: {response.text}")
            return []
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

def weaviate_get_class_documents(cls):
    PORT = 8000
    class_name = cls
    data_type = 'Weaviate'
    mode='get_all_document_per_class'
    POST_URL = f"http://localhost:{PORT}/VectoreDataBase/?data_type={data_type}&class_name={class_name}&mode={mode}"
    response = requests.post(POST_URL)
    if response.status_code == 200:
        try:
            weaviate_data = response.json()
            weaviate_list = weaviate_data.get("weaviate", [])
            print('weaviate list', weaviate_list)
            return weaviate_list
        except requests.exceptions.JSONDecodeError:
            print(f"Failed to decode JSON. Response content: {response.text}")
            return []
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

st.title("Vector Database Management Dashboard")

# Fetch and display collections
st.sidebar.header("Collections")

# Display documents in the main area
st.header("Documents")

weaviate_row1_col1, weaviate_row1_col2 = st.columns(2)

with weaviate_row1_col1:
    st.subheader('Create a class')
    class_name = st.text_input("Enter class name:")
    class_name = class_name.replace(" ", "-")
    class_description = st.text_input("Enter a class description (optional):")
    class_vectorizer = st.selectbox("Select vectorizer:", ['text2vec-transformers', "Instructor-XL", "BERT"])
    if st.button("Create Class"):
              if class_name is not None:
                    print('class name', class_name)
                    data_type = "Weaviate"
                    mode = "create_class"
                    PORT_NUMBER = 8000
                    POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&class_name={class_name}&embedding_name={class_vectorizer}&description={class_description}&mode={mode}"
                    response = requests.post(POST_URL)
                    st.success(f"Class {class_name} added!")
                    weaviate_classes = weaviate_get_classes()



    st.subheader("Delete Class")
    remove_class_name = st.text_input("Enter class to delete:")
    remove_class_name = remove_class_name.replace(" ", "-")
    if st.button("Delete Class"):
        if class_name is not None:
                    print('class name', remove_class_name)
                    data_type = "Weaviate"
                    mode = "delete_class"
                    PORT_NUMBER = 8000
                    POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&class_name={remove_class_name}&mode={mode}"
                    response = requests.post(POST_URL)
                    st.success(f"Class {remove_class_name} deleted!")
                    weaviate_classes = weaviate_get_classes()

with weaviate_row1_col2:
    weaviate_classes = weaviate_get_classes()
    st.table(weaviate_classes)

weaviate_row2_col1, weaviate_row2_col2 = st.columns(2)

with weaviate_row2_col1:
       ########## Create a class ##########
        st.subheader("Select class to add documents to")
        weaviate_classes = weaviate_get_classes()
        class_name = st.selectbox("Select class:", weaviate_classes)
        document_type = st.radio("Select document type:", ('Webpage', 'PDF'))
        if document_type == 'Webpage':
            weaviate_webpage_uploader = st.text_input("Enter webpage URL:")
            weaviate_webpage_name = st.text_input("Enter webpage name:", key='webpage_name')
            if st.button("Add webpage to class"):
                    if weaviate_webpage_uploader is not None:
                        data_type = "Weaviate"
                        mode = "add_webpage"
                        PORT_NUMBER = 8000
                        POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&doc_name={weaviate_webpage_name}&data_path={weaviate_webpage_uploader}&collection={class_name}&mode={mode}"
                        response = requests.post(POST_URL)
                        st.success(f"Webpage {weaviate_webpage_uploader} added to class {class_name}!")
                        weaviate_classes = weaviate_get_classes()

        elif document_type == 'PDF':
            document_name = st.text_input("Enter document name:") #TODO add control for empty strings and such.
            st.subheader("Add document to the class")
            weaviate_uploaded_pdf = st.file_uploader("Weaviate add PDF",type=['pdf'])
            #class_to_add_document = st.text_input("Enter PDF name:")
            #class_to_add_document = class_to_add_document.replace(" ", "-")
            if st.button("Add document to class"):
                    if weaviate_uploaded_pdf is not None:
                        weaviate_pdf_filename = os.path.join(pdf_directory, weaviate_uploaded_pdf.name)
                        with open(weaviate_pdf_filename, "wb") as f:
                                f.write(weaviate_uploaded_pdf.read())
                        print('pdf filename', weaviate_pdf_filename)
                        #print('class name', class_to_add_document)
                        data_type = "Weaviate"
                        mode = "add_pdf"
                        PORT_NUMBER = 8000
                        print('class name', class_name)
                        POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&document_name={document_name}&pdf_path={weaviate_pdf_filename}&class_name={class_name}&mode={mode}"
                        response = requests.post(POST_URL)
                        st.success(f"Document {weaviate_pdf_filename} added to class {class_name}!")
                        weaviate_classes = weaviate_get_classes()

        st.subheader("Remove document from class")
        weaviate_document_to_remove = st.text_input("Enter document name:", key='remove_doc_weaviate')
        if st.button("Remove document from class"):
            if weaviate_document_to_remove is not None:
                    data_type = "Weaviate"	
                    mode = "delete_document"
                    PORT_NUMBER = 8000
                    POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&document_name={weaviate_document_to_remove}&class_name={class_name}&mode={mode}"
                    response = requests.post(POST_URL)
                    st.success(f"Document {weaviate_document_to_remove} removed from class {class_name}!")

with weaviate_row2_col2:
    st.subheader(f"Document available in {class_name}")
    class_documents = weaviate_get_class_documents(class_name)
    print('class documents', class_documents)
    st.table(class_documents)


