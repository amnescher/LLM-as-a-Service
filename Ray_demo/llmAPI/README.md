export CONFIG_PATH=/path/to/your/config.yaml
cd API
uvicorn app:app --reload --port 8080

ray start --head
serve build backend:app -o config.yaml
serve deploy config.yaml
