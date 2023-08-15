install required packages
```
pip install -r requirement.txt
```
Store your Huggingface access token in the env.env file and save it in the same directory. 
e.g., content of en.env file.
Hugging_ACCESS_TOKEN = xxxxxxxx

run frontend 
```
streamlit run frontend_RAY.py
```

Generate a multi-application config file 

```
serve build  ray_backend:app ray_VectorDB:app -o  config.yaml
```

start a Ray cluster

```
ray start --head
```
deploy the applications

```
serve deploy config.yaml
```
