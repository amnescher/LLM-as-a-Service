from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from ts.torch_handler.base_handler import BaseHandler
import logging
import zipfile

logger = logging.getLogger(__name__)

class FalconHandler(BaseHandler):
    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir + "/model")

        self.model_name = model_dir + "/model"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            local_files_only=True,
            trust_remote_code=True,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": 0},
        )

    def preprocess(self, data):
        return data

    def inference(self, requests):

        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")

        sequences = self.pipeline(
            input_text,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        results = []
        for seq in sequences:
            results.append(seq["generated_text"])
        return results

    def postprocess(self, data):
        return data

