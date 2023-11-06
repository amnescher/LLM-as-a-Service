from ray.job_submission import JobSubmissionClient
runtime_env = {"working_dir": "/home/ubuntu/LLM/LLM-as-a-Service/Ray_demo/llmAPI", "pip":  [
        "langchain",
        "fastapi",
        "wandb",
        "ray[data,train,tune,serve]",
        "eschercloud",
        "SQLAlchemy",
        "passlib",
        "-f https://download.pytorch.org/whl/cu118 torch",
        "-f https://download.pytorch.org/whl/cu118 torchvision",
        "-f https://download.pytorch.org/whl/cu118 torchaudio",
        "xformers",
        "accelerate",
        "sentencepiece",
        "bitsandbytes",
        "einops",
        "streamlit",
        "python-dotenv",
        "transformers",
        "scipy"
    ]}
client = JobSubmissionClient("http://192.168.199.207:8265")
job_id = client.submit_job(
    entrypoint="python backend.py",
    runtime_env=runtime_env,
)