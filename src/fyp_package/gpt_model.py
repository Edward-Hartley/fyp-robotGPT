# Wrapper for using chatgpt through openai's API
from openai import OpenAI
from openai import RateLimitError, APIConnectionError
from time import sleep
import os
import json
import base64
from fyp_package import config, utils
import base64

def get_api_key():
    # openai api key in .openai_key file
    with open('.openai_key', 'r') as f:
        return f.readline().strip()
    
class GptModel:
    def __init__(self,
                 model=config.default_openai_model,
                 tools=None,
                 temperature=config.model_temperature,
                 max_tokens=1000,
                 stop=None,
                 client=None
                ):
        if not client:
            self.client = OpenAI(api_key=get_api_key())
        else:
            self.client = client
        self.model = model
        self.tools = tools
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop

    def chat_completion(self, messages, tool_choice="auto"):

        while True:
            try:
                completion = self.chat_completion_request(messages, tool_choice)
                break
            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)

        return completion

    def chat_completion_request(self, messages, tool_choice="auto"):
        if self.tools is None:
            tool_choice = None

        message = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            tools=self.tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.stop,
            tool_choice=tool_choice
        ).choices[0].message

        return message.content.strip()
    
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

def text_content(text):
    return {
        "type": "text",
        "text": text
    }

def build_message(text, role):
    return ({
        'role': role,
        'content': text
    })

def build_tool_response(text, tool_call_id):
    return ({
        'role': 'tool',
        'content': text,
        'tool_call_id': tool_call_id
    })

def build_image_message(image_path, detail="low", text=None):
    content = [image_content(image_path, detail)]
    if text:
        content.append(text_content(text))
    return ({
        'role': 'user',
        'content': content
    })

def build_image_tool_response(image_path, tool_call_id, detail="low", text=None):
    '''
    Returns two messages: one to respond to the tool and let the model know that the image will be in the following user message
    '''
    return build_tool_response("Image response", tool_call_id), build_image_message(image_path, detail, text)


if __name__ == '__main__':
    from fyp_package.experiments.CAP_vision_agent_prompts import *
    gpt = GptModel(model=config.cheap_openai_model, tools=[code_tool, return_tool])
    messages = []
    messages.append(build_message("You have access to two tools that let you execute python code on a tabletop system and return variables from that code to the user (you can also return new explicit values, the returned string will be 'eval'ed on the tabletop system). Please always reason about your chain of thought _before_ using any tools.", "system"))
    messages.append(build_message("Hello!", 'user'))
    completion = gpt.chat_completion(messages)
    print(completion)