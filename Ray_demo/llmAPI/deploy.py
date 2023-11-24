from backend import PredictDeployment
from vector_database import VectorDataBase
from ray import serve
import yaml



class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


with open("cluster_conf.yaml", 'r') as file:
    config = yaml.safe_load(file)
    config = Config(**config)

serve.run(VectorDataBase.bind(), name = "VectorDB" ,route_prefix="/VectorDB")

for LLM in config.LLMs:
    prefix = LLM["route_prefix"]
    model_id = LLM["model_id"]
    # Define the deployment with the specified configurations from LLM
    deployment_options = {
        "name": LLM["name"],
        "route_prefix": f"/{prefix}",
        "ray_actor_options": {"num_gpus": LLM["num_gpus"]},
        "autoscaling_config": {
            "min_replicas": LLM["min_replicas"],
            "initial_replicas": LLM["initial_replicas"],
            "max_replicas": LLM["max_replicas"],
            "target_num_ongoing_requests_per_replica": LLM["target_num_ongoing_requests_per_replica"],
            "graceful_shutdown_wait_loop_s": LLM["graceful_shutdown_wait_loop_s"],
            "max_concurrent_queries": LLM["max_concurrent_queries"],
        }
    }
    # Apply the deployment options to the PredictDeployment class
    PredictDeployment = PredictDeployment.options(**deployment_options)
    # Deploy the service with the specified model_id and other parameters
    serve.run(PredictDeployment.bind(model_id, LLM["temperature"], LLM["max_new_tokens"], LLM["repetition_penalty"], LLM["batch_size"]), name = LLM["name"], route_prefix=f"/{prefix}")