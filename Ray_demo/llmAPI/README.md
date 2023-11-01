export CONFIG_PATH=/path/to/your/cluster_conf.yaml
cd API
uvicorn app:app --reload --port 8080

ray start --head
serve build backend:app -o config.yaml
serve deploy config.yaml
