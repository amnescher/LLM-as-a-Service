import binascii
from typing import Any, List

import pypdf
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
import io
from ray.data.datasource import FileExtensionFilter
import weaviate
from langchain.vectorstores import Weaviate
from langchain.text_splitter import CharacterTextSplitter
import yaml

# ---------------------Document Loading
class HFEmbeddings:
    def __init__(self):
        self.embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"})
    def __call__(self, text_batch: List[str]):
        embeddings = self.embedding.embed_query(text_batch)
        return list(zip(text_batch, embeddings))

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


with open("vdb_conf.yaml", "r") as file:
    config = yaml.safe_load(file)
    config = Config(**config)

@serve.deployment(ray_actor_options={"num_gpus": 0.2}, autoscaling_config={
        "min_replicas": config.min_replicas,
        "initial_replicas": config.initial_replicas,
        "max_replicas": config.max_replicas,
        "max_concurrent_queries": config.max_concurrent_queries,}, route_prefix="/VectoreDataBase")

class VectorDataBase:
    def __init__(self):
        self.embed = HFEmbeddings()

        self.weaviate_client = weaviate.Client(
            url="http://localhost:8080",   
        )
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

        self.vectorstore_video = Chroma(
            "YouTube_store",
            self.embeddings,
            persist_directory=self.video_persist_directory,
        )
        self.vectorstore_video.persist()

        self.weaviate_vectorstore = Weaviate(self.weaviate_client, 'Chatbot', 'page_content', attributes=['page_content'])

    def load_processed_files(self):
        if os.path.exists("processed_files.json"):
            with open("processed_files.json", "r") as f:
                return json.load(f)
        else:
            return []

    def is_file_processed(self, filename):
        processed_files = self.load_processed_files()
        return filename in processed_files
        
    def save_processed_file(self, filename):
        processed_files = self.load_processed_files()
        processed_files.append(filename)
        with open("processed_files.json", "w") as f:
            json.dump(processed_files, f)

    def add_video_to_DB(self, url, collection,id):
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        result = loader.load()
        if id == "" or id == " " or id == None:
            id= result[0].metadata['title']
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

    def weaviate_serialize_document(self, doc, title):
        
        return {
            "page_content": doc.page_content,
            "document_title": title,
        }

    def create_weaviate_class(self, name, embedding_model):
        class_name = str(name)
        #class_description = str(description)
        vectorizer = str(embedding_model)
        
        schema = {'classes': [ 
            {
                    'class': class_name,
                    'description': 'normal description',
                    'vectorizer': vectorizer,
                    'moduleConfig': {
                        vectorizer: {
                            'vectorizerClassName': False,
                            }
                    },
                    'properties': [{
                        'dataType': ['text'],
                        'description': 'the text from the documents parsed',
                        'moduleConfig': {
                            vectorizer: {
                                'skip': False,
                                'vectorizePropertyName': False,
                                }
                        },
                        'name': 'page_content',
                    },
                    {
                        'name': 'document_title',
                        'dataType': ['text'],
                    }],      
                    },
        ]}
        self.weaviate_client.schema.create(schema)

    ############ Test Ray Function for embedding ############
    def process_document_batch_ray(self, docs, collection_name, doc_name):
        #df = ray.data.read_binary("../pdf_querying_summarization/temp_file/doc.pdf", FileExtensionFilter(".pdf"))
        #df = df.flat_map(self.convert_to_text)
        #df = df.flat_map(self.split_text)
        df = df.map_batches(
            self.adding_weaviate_document(),
            compute = ray.data.ActorPollStrategy(min_size=1, max_size=2),
            num_gpus=0.1
        )
        print('process function called')

    def convert_to_text(self,pdf_bytes: bytes):
        pdf_bytes_io = io.BytesIO(pdf_bytes)

        try:
            pdf_doc = pypdf.PdfReader(pdf_bytes_io)
        except pypdf.errors.PdfStreamError:
            # Skip pdfs that are not readable.
            # We still have over 30,000 pages after skipping these.
            return []

        text = []
        for page in pdf_doc.pages:
            try:
                text.append(page.extract_text())
            except binascii.Error:
                # Skip all pages that are not parseable due to malformed characters.
                print("parsing failed")
        return text

    def split_text(self, page_text: str):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, length_function=len
        )
        split_text: List[str] = text_splitter.split_text(page_text)

        split_text = [text.replace("\n", " ") for text in split_text]
        return split_text



    def adding_weaviate_document(self, docs, collection_name, doc_name):
        loader = PyPDFLoader(docs)

        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        text_docs = text_splitter.split_documents(documents)
        serialized_docs = [
            self.weaviate_serialize_document(doc,doc_name) 
            for doc in text_docs
            ]
        self.weaviate_client.batch.configure(batch_size=50, num_workers=10, dynamic=True)

        with self.weaviate_client.batch as batch:
            for text in serialized_docs:
                    batch.add_data_object(
                        text,
                        class_name=collection_name, 
        )

    def adding_weaviate_webpage(self, url, collection_name, doc_name):
        # Load content from the webpage
        loader = WebBaseLoader(str(url))
        result = loader.load()
        
        # Extracting the ID (title) if not provided
        if not doc_name or doc_name.isspace():
            doc_name = result[0].metadata['title']
            
        # Split the content into manageable chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(result)
        
        # Serialize the documents for Weaviate
        # Assuming the function weaviate_serialize_document serializes the document for Weaviate
        serialized_docs = [self.weaviate_serialize_document(doc, doc_name) for doc in texts]
        
        # Configure batch and add to Weaviate
        self.weaviate_client.batch.configure(batch_size=50)
        with self.weaviate_client.batch as batch:
            for text in serialized_docs:
                batch.add_data_object(
                    text,
                    class_name=collection_name
                )

    def delete_weaviate_class(self, name):
        class_name = name
        self.weaviate_client.schema.delete_class(class_name)

    def delete_weaviate_document(self, name, cls_name):
        document_name = str(name)
        self.weaviate_client.batch.delete_objects(
            class_name=cls_name,
            where={
                "path": ["document_title"],
                "operator": "Like",
                "valueText": document_name,
            }
        )

    def query_weaviate_document_names(self,cls):
        class_properties = ["document_title"]
        query = self.weaviate_client.query.get(cls, class_properties)
        query = query.do()

        document_title_set = set()
        documents = query.get('data', {}).get('Get', {}).get(str(cls), [])

        for document in documents:
            document_title = document.get('document_title')
            if document_title is not None:
                document_title_set.add(document_title)
        return list(document_title_set)

    def get_embeddings(self, text):
        #string_text = [doc.page_content for doc in text]
        embeddings_vec = self.embeddings.embed_query(text)
        return embeddings_vec

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
        elif data_type == "Weaviate":
            if mode == "create_class":
                class_name = request.query_params["class_name"]	
                embedding_name = request.query_params["embedding_name"]
                #description = request.query_params["description"]
                #class_description = request.query_params["class_description"]
                self.create_weaviate_class(class_name, embedding_name)
            elif mode == "get_all":
                classes = self.weaviate_client.schema.get()
                class_names = [cls['class'] for cls in classes['classes']]
                return JSONResponse(content={"weaviate": class_names})
            elif mode == "delete_class":
                class_name = request.query_params["class_name"]
                self.delete_weaviate_class(class_name)
            elif mode == "add_pdf":
                pdf_path = request.query_params["pdf_path"]
                class_name = request.query_params["class_name"]
                document_name = request.query_params["document_name"]
                self.adding_weaviate_document(pdf_path, class_name, document_name)
            elif mode == "add_webpage":
                page_name = request.query_params['doc_name']
                collection = request.query_params['collection']
                page_url = request.query_params['data_path']
                self.adding_weaviate_webpage(page_url, collection, page_name)
            elif mode == "get_all_document_per_class":
                cls = request.query_params["class_name"]
                class_documents = self.query_weaviate_document_names(cls)
                return JSONResponse(content={"weaviate": class_documents})
            elif mode == "delete_document":
                document_name = request.query_params["document_name"]
                cls = request.query_params["class_name"]
                self.delete_weaviate_document(document_name, cls)
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