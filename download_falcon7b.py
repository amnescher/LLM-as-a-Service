import urllib.request

file_urls = [
    "https://huggingface.co/tiiuae/falcon-7b-instruct/raw/main/config.json",
    "https://huggingface.co/tiiuae/falcon-7b-instruct/raw/main/configuration_RW.py",
    "https://huggingface.co/tiiuae/falcon-7b-instruct/raw/main/generation_config.json",
    "https://huggingface.co/tiiuae/falcon-7b-instruct/raw/main/modelling_RW.py",
    "https://huggingface.co/tiiuae/falcon-7b-instruct/resolve/main/pytorch_model-00001-of-00002.bin",
    "https://huggingface.co/tiiuae/falcon-7b-instruct/resolve/main/pytorch_model-00002-of-00002.bin",
    "https://huggingface.co/tiiuae/falcon-7b-instruct/raw/main/pytorch_model.bin.index.json",
    "https://huggingface.co/tiiuae/falcon-7b-instruct/raw/main/special_tokens_map.json",
    "https://huggingface.co/tiiuae/falcon-7b-instruct/raw/main/tokenizer.json",
    "https://huggingface.co/tiiuae/falcon-7b-instruct/raw/main/tokenizer_config.json"
]

path = "/home/ubuntu/falcon/kubeflow/llm-falcon-chatbot/weights_7b/"  # Path provided by the user

for url in file_urls:
    file_name = url.split("/")[-1]  # Extract the file name from the URL
    file_path = path + file_name  # Construct the file path using the provided path and file name
    urllib.request.urlretrieve(url, file_path)  # Download the file