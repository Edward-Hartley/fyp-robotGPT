# Wrapper for using chatgpt through openai's API
from openai import OpenAI
import os
import json
import base64
from fyp_package import config, prompts
import base64

def get_api_key():
    # openai api key in .openai_key file
    with open('.openai_key', 'r') as f:
        return f.readline().strip()
    
# encode image to base64
# image path must point to either png or jpg file
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def image_content(image_path, detail="auto"):
    image_type = image_path.split(".")[-1]
    if image_type == "jpg":
        image_type = "jpeg"
    if image_type not in ["png", "jpeg"]:
        raise ValueError("Image type must be either png, jpg or jpeg")
    
    base64_image = encode_image(image_path)

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/{image_type};base64,{base64_image}",
            "detail": detail
        }
    }

class OneRobotAssistant:
    def __init__(self, model=config.default_openai_model, function_tools=[]):
        self.api_key = get_api_key()
        self.client = OpenAI(self.api_key)
        self.model = model
        self.assistant = self.client.beta.assistants.create(
            model=model,
            name="OneRobotAssistant",
            description="Robot control in a single assistant",
            instructions=prompts.oneRobotAssistantInstructions,
            tools=[{"type": "code_interpreter"}] + function_tools,
            tool_resources=None,
            temperature=config.model_temperature
        )
        self.thread = None

    def create_thread(self, messages=None, extra_files=None):
        if messages is None:
            thread = self.client.beta.threads.create()

    def add_file_to_thread_code_interpreter(self, file_id):
        existing_tool_resources = self.get_code_interpreter_files()

        self.client.beta.assistants.update(
            self.assistant.id,
            tool_resources={
                "code_interpreter": {"file_ids": existing_tool_resources.append(file_id)},
                "file_search": {"vector_store_ids": []},
            }
        )

    def get_thread_code_interpreter_files(self):
        if self.assistant.tool_resources is None:
            return []
        else:
            return [self.assistant.tool_resources.code_interpreter]
        
    def upload_file(self, file_path):
        with open(file_path, "rb") as file:
            file_id = self.client.files.create(
                file=file,
                purpose="assistants"
                )
        
        return file_id
    
    def upload_image(self, file_path):
        with open(file_path, "rb") as file:
            file_id = self.client.files.create(
                file=file,
                purpose="vision"
                )
        
        return file_id


    