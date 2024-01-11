import os
import asyncio
from typing import Any
from typing import AsyncIterable
import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from langchain.chains import LLMChain
app = FastAPI()
class Message(BaseModel):
    content: str
# initialize the agent (we need to do this for the callbacks)
from langchain_community.llms import HuggingFaceTextGenInference
from langchain import PromptTemplate


llm = HuggingFaceTextGenInference(
    inference_server_url="http://localhost:8082/",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    streaming=True,
)
template = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.Chat History:\n\n{chat_history} \n\nUser: {question}"
my_prompt = PromptTemplate(
            input_variables=["chat_history", "question"], template=template
        )
llm_chain =LLMChain(llm=llm, prompt=my_prompt , verbose=False, output_key="output")



# async def send_message(content: str) -> AsyncIterable[str]:
#     callback = AsyncIteratorCallbackHandler()
#     llm.callbacks = [callback]

#     task = asyncio.create_task(
#         llm.agenerate(content)
#     )

#     try:
#         async for token in callback.aiter():
#             yield token
#     except Exception as e:
#         print(f"Caught exception: {e}")
#     finally:
#         callback.done.set()
#     await task


# @app.post("/stream_chat/")
# async def stream_chat(message: Message):
#     generator = send_message(message.content)
#     return StreamingResponse(generator, media_type="text/event-stream")
    
