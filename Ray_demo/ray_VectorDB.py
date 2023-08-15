import ray
from langchain.chains.conversation.memory import ConversationBufferMemory
import pandas as pd
from ray import serve
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from starlette.requests import Request
from langchain.vectorstores import Chroma
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from langchain.document_loaders import PyPDFLoader

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
            os.makedirs(self.ideo_persist_directory)
            print(f"Directory '{self.video_persist_directory}' created successfully.")
        self.vectorstore_doc = Chroma(
            "PDF_store", self.embeddings, persist_directory=self.Doc_persist_directory
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

    def add_video_to_DB(self, url):
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        result = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=400
        )
        texts = text_splitter.split_documents(result)
        self.vectorstore_video.add_documents(texts)
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

    def save_video(self, video_url):
        if os.path.exists("processed_videos.json"):
            with open("processed_videos.json", "r") as f:
                video_list = json.load(f)
                if video_url not in video_list:
                    self.add_video_to_DB(video_url)
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
                self.add_video_to_DB(video_url)
            return True

    async def __call__(self, request):
        data_type = request.query_params["data_type"]
        mode = request.query_params["mode"]
        if data_type == "Video":
            if mode == "clear":
                self.clear_videos()
            else:
                video_url = request.query_params["data_path"]
                print("recieved Video ----->", video_url)
                self.save_video(video_url)
                print("video Uploaded Successfully")
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
