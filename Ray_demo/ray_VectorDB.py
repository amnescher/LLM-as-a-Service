import ray
from langchain.chains.conversation.memory import ConversationBufferMemory
import pandas as pd
from ray import serve
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from langchain.vectorstores import Chroma
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from langchain.document_loaders import PyPDFLoader, WebBaseLoader

# ---------------------Document Loading


@serve.deployment(ray_actor_options={"num_gpus": 0.1}, route_prefix="/VectoreDataBase")
class VectorDataBase:
    def __init__(self):
        self.Doc_persist_directory = "./Document_db"
        self.video_persist_directory = "./YouTube_db"

        self.embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"}
        )
        # Check if DB1 directory exists, and if not, create it
        if not os.path.exists(self.Doc_persist_directory):
            os.makedirs(self.Doc_persist_directory)
            print(f"Directory '{self.Doc_persist_directory}' created successfully.")

        # Check if DB2 directory exists, and if not, create it
        if not os.path.exists(self.video_persist_directory):
            os.makedirs(self.video_persist_directory)
            print(f"Directory '{self.video_persist_directory}' created successfully.")
        self.vectorstore_doc = Chroma(
            persist_directory=self.Doc_persist_directory, embedding_function=self.embeddings 
        )
        self.vectorstore_doc.persist()
        self.vectorstore_video = Chroma(
            "YouTube_store",
            self.embeddings,
            persist_directory=self.video_persist_directory,
        )
        self.vectorstore_video.persist()

    def load_processed_files(self):
        if os.path.exists("processed_files.json"):
            with open("processed_files.json", "r") as f:
                return json.load(f)
        else:
            return []

    def is_file_processed(self, filename):
        processed_files = self.load_processed_files()
        return filename in processed_files
        
    def add_pdf_to_DB(self, pdf_path):
        if not self.is_file_processed(pdf_path):
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=200
            )
            texts = text_splitter.split_documents(pages)
            self.vectorstore_doc.add_documents(texts)
            self.vectorstore_doc.persist()
            return True
        else:
            return False

    def save_processed_file(self, filename):
        processed_files = self.load_processed_files()
        processed_files.append(filename)
        with open("processed_files.json", "w") as f:
            json.dump(processed_files, f)

    def add_video_to_DB(self, url, collection,id):
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        result = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=400
        )
        texts = text_splitter.split_documents(result)
        selected_collection = self.vectorstore_doc._client.get_collection(collection)
        selected_collection.add(
            documents = [doc.page_content for doc in texts],
            embeddings=[self.get_embeddings(text) for text in [doc.page_content for doc in texts]],
            #metadatas = [{'test32':"test"+str(i)} for i in range(1, len(texts) + 1)],
            ids=[(str(id)+ "_" +str(i)) for i in range(1, len(texts) + 1)],
        )
        #self.vectorstore_doc.add_documents(texts)
        self.vectorstore_doc.persist()

    def clear_videos(self):
        self.vectorstore_video.delete([])
        self.vectorstore_video.persist()
        with open("processed_videos.json", "w") as f:
            json.dump([], f)

    def clear_docs(self):
        self.vectorstore_doc.delete([])
        with open("processed_files.json", "w") as f:
            json.dump([], f)

    def save_video(self, video_url, collection, id):
        if os.path.exists("processed_videos.json"):
            with open("processed_videos.json", "r") as f:
                video_list = json.load(f)
                if video_url not in video_list:
                    self.add_video_to_DB(video_url, collection, id)
                    video_list.append(video_url)
                    with open("processed_videos.json", "w") as f:
                        json.dump(video_list, f)
                    return True
                else:
                    return False
        else:
            print("new video")
            # If the file doesn't exist, create it and add the first video URL
            video_list = [video_url]
            with open("processed_videos.json", "w") as f:
                json.dump(video_list, f)
                self.add_video_to_DB(video_url, collection, id)
            return True

    def create_new_collection(self, name):
        final_name = str(name)
        print('final name', final_name)
        collection = self.vectorstore_doc._client.get_or_create_collection(final_name)
        print('collection created? ', collection)

    def remove_collection(self, name):
        final_name = str(name)
        collection_to_remove = self.vectorstore_doc._client.get_collection(final_name)
        collection_to_remove.delete()
        self.vectorstore_doc._client.delete_collection(final_name)

    def get_all_collections(self):
        collection_list = self.vectorstore_doc._client.list_collections()
        collection_names = [collection.name for collection in collection_list]
        return collection_names
    
    def get_unique_ids(self, collection):
        selected_collection = self.vectorstore_doc._client.get_collection(collection)
        all_ids = selected_collection.get()['ids']
        id_prefixe = []
        for id in all_ids:
            prefix = id.split('_')[0]
            if prefix not in id_prefixe:
                id_prefixe.append(prefix)
        return id_prefixe

    def delete_document(self, collection, name):
        final_col_name = str(collection)
        final_doc_name = str(name)
        selected_collection = self.vectorstore_doc._client.get_collection(final_col_name)
        document_filter = selected_collection.get()['ids']
        filtered_ids = [id for id in document_filter if id.startswith(final_doc_name)]
        selected_collection.delete(
            ids = filtered_ids
        )

    def get_embeddings(self, text):
        #string_text = [doc.page_content for doc in text]
        embeddings_vec = self.embeddings.embed_query(text)
        return embeddings_vec
    
    def adding_webpage(self, url, collection, id):
        loader= WebBaseLoader(str(url))
        result = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=400
        )
        texts = text_splitter.split_documents(result)
        selected_collection = self.vectorstore_doc._client.get_collection(collection)
        selected_collection.add(
            documents = [doc.page_content for doc in texts],
            embeddings=[self.get_embeddings(text) for text in [doc.page_content for doc in texts]],
            ids=[(str(id)+ "_" +str(i)) for i in range(1, len(texts) + 1)],
        )
        self.vectorstore_doc.persist()

    def adding_documents(self, pdf, collection, id):
        loader = PyPDFLoader(pdf)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
        texts = text_splitter.split_documents(pages)
        selected_collection = self.vectorstore_doc._client.get_collection(collection)
        selected_collection.add(
            documents = [doc.page_content for doc in texts],
            embeddings=[self.get_embeddings(text) for text in [doc.page_content for doc in texts]],
            metadatas = [{'test32':"test"+str(i)} for i in range(1, len(texts) + 1)],
            ids=[(str(id)+ "_" +str(i)) for i in range(1, len(texts) + 1)],
        )
        selected_collection.get()['ids']   

    async def __call__(self, request):
        data_type = request.query_params["data_type"]
        mode = request.query_params["mode"]
        if data_type == "Video":
            if mode == "clear":
                self.clear_videos()
            else:
                video_url = request.query_params["data_path"]
                collection = request.query_params['collection']
                video_name = request.query_params['doc_name']
                print("recieved Video ----->", video_url)
                self.save_video(video_url,collection, video_name)
                print("video Uploaded Successfully")
        elif data_type == "Collection":
            if mode == "remove":
                removed_collection = request.query_params["data_path"]
                self.remove_collection(removed_collection)
            elif mode == 'get_all':
                collections = self.get_all_collections()
                return JSONResponse(content={"collections": collections})
                #response = Response(collections, media_type='text/plain')
            elif mode == "add":
                collection_name = request.query_params["data_path"]
                print('adding a collection', collection_name)
                self.create_new_collection(collection_name)
            else: 
                print('nothing/....')
        elif data_type == "Webpage":
            if mode == "add":
                page_name = request.query_params['doc_name']
                collection = request.query_params['collection']
                page_url = request.query_params['data_path']
                self.adding_webpage(page_url, collection, page_name)
        elif data_type == "Document":
            if mode == "add":
                pdf = request.query_params["data_path"]
                collection = request.query_params["collection"]
                doc_name = request.query_params["doc_name"]
                self.adding_documents(pdf,collection,doc_name)
            elif mode == "get_all":
                collection_name = request.query_params['collection']
                documents_in_col = self.get_unique_ids(str(collection_name))
                #print('the type of the data', type(documents_in_col), documents_in_col)
                return JSONResponse(content={"collection": documents_in_col})
            elif mode == "remove":
                collection_name = request.query_params['remove']
                document_name = request.query_params['doc']
                self.delete_document(collection_name, document_name)

        else:
            if mode == "clear":
                self.clear_docs()
            else:
                file_path = request.query_params["data_path"]
                print("Pdf Path---->", file_path)
                self.add_pdf_to_DB(file_path)
                self.save_processed_file(file_path)
                os.remove(file_path)

app = VectorDataBase.bind()