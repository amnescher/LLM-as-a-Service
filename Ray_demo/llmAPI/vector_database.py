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
import time
from langchain.document_loaders import TextLoader


@ray.remote(num_gpus=0.1)
class WeaviateEmbedder:
    def __init__(self):
        self.time_taken = 0
        self.text_list = []
        self.weaviate_client = weaviate.Client(
            url="http://localhost:8080",   
        )

    def adding_weaviate_document(self, text_lst, collection_name):
        start_time = time.time()
        self.weaviate_client.batch.configure(batch_size=50)

        with self.weaviate_client.batch as batch:
            for text in text_lst:
                    batch.add_data_object(
                        text,
                        class_name=collection_name, 
                        #uuid=generate_uuid5(text),
        )
        self.text_list.append(text)
        self.time_taken = time.time() - start_time
        return self.text_list

    def get(self):
        return self.lst_embeddings
    
    def get_time_taken(self):
        return self.time_taken
    
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
    config = Config(**config)

@serve.deployment(ray_actor_options={"num_gpus": 0.4}, autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 3,
        "max_concurrent_queries": 2,})



class VectorDataBase:
    def __init__(self):

        self.weaviate_client = weaviate.Client(
            url="http://localhost:8080",   
        )
        self.weaviate_vectorstore = Weaviate(self.weaviate_client, 'Chatbot', 'page_content', attributes=['page_content'])

    def weaviate_serialize_document(self,doc):
        document_title = doc.metadata.get('source', '').split('/')[-1]
        return {
            "page_content": doc.page_content,
            "document_title": document_title,
        }
    
    def weaviate_split_multiple_pdf(self,docs):    
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        text_docs = text_splitter.split_documents(docs)

        serialized_docs = [
                    self.weaviate_serialize_document(doc) 
                    for doc in text_docs
                    ]
        return serialized_docs	

    def divide_workload(self, num_actors, documents):
        docs_per_actor = len(documents) // num_actors

        doc_parts = [documents[i * docs_per_actor: (i + 1) * docs_per_actor] for i in range(num_actors)]

        if len(documents) % num_actors:
            doc_parts[-1].extend(documents[num_actors * docs_per_actor:])

        return doc_parts

    def parse_pdf(self, directory):    
        documents = []
        for file in os.listdir(directory):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(directory, file)
                try:
                    loader = PyPDFLoader(pdf_path)
                    documents.extend(loader.load())
                except pypdf.errors.PdfStreamError as e:
                    print(f"Skipping file {file} due to error: {e}")
                    continue  # Skip this file and continue with the next one
            elif file.endswith('.txt'):
                text_path = os.path.join(directory, file)
                try:
                    loader = TextLoader(text_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error in file {file}: {e}")
                    continue
        return documents

    def process_all_docs(self, dir, cls):
        document_list = self.parse_pdf(dir)
        serialized_docs = self.weaviate_split_multiple_pdf(document_list)
        if len(serialized_docs) <= 50:
            self.add_weaviate_document(cls, serialized_docs)
        else:
            doc_workload = self.divide_workload(3, serialized_docs)
            self.add_weaviate_batch_documents(cls, doc_workload)

    def add_weaviate_document(self, cls, docs):
        actor = WeaviateEmbedder.remote()
        [actor.adding_weaviate_document.remote(doc_part, str(cls)) for doc_part in docs]

    def add_weaviate_batch_documents(self, cls, doc_workload):
        actors = [WeaviateEmbedder.remote() for _ in range(3)]
        [actor.adding_weaviate_document.remote(doc_part, str(cls)) for actor, doc_part in zip(actors, doc_workload)]
        [actor.get_time_taken.remote() for actor in actors]

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

    def delete_weaviate_class(self, name):
        class_name = name
        self.weaviate_client.schema.delete_class(class_name)

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
                self.process_all_docs(pdf_path, class_name)
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

serve.run(VectorDataBase.bind(), route_prefix="/VectoreDataBase")
#app = VectorDataBase.bind()