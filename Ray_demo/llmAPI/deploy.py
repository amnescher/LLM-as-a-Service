from backend import PredictDeployment
from vector_database import VectorDataBase
from ray import serve

serve.run(VectorDataBase.bind(), name = "VectorDB" ,route_prefix="/VectorDB")
#serve.run(PredictDeployment.bind(), name="LLM", route_prefix="/llm")