# Wrapper for using chatgpt through openai's API
from openai import OpenAI
from openai import RateLimitError, APIConnectionError
from time import sleep
import os
import json
import base64
from fyp_package import config, utils
import base64
import cv2

from fyp_package.experiments import agent_logging

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
                 client=None,
                 name=None
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
        self.name = name
        self.most_recent_usage = None

    @agent_logging.log_chat_completion
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
        kwargs = {
            'messages': messages,
            'model': self.model,
            'tools': self.tools,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }

        if self.tools is not None:
            kwargs['tool_choice'] = tool_choice

        if self.stop is not None:
            kwargs['stop'] = self.stop

        completion = self.client.chat.completions.create(**kwargs)
        self.most_recent_usage = completion.usage

        return completion.choices[0].message.content.strip()
    
# encode image to base64
# image path must point to either png or jpg file
def encode_image(image_path):
    if not config.simulation:
        # load image and crop it from the left by config.view_image_crop_left
        image = cv2.imread(image_path)
        image = image[:, config.view_image_crop_left:]
        cv2.imwrite(config.temp_image_path, image)
        image_path = config.temp_image_path

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
    agent_logging.setup_logging()
    gpt = GptModel(model=config.cheap_openai_model, name='test')
    messages = []

    message = '''
    Hello please can you just repeat the below text exactly, character for character, including any punctuation, capitalisation, and line breaks:
    <text begin>
    ##$$**



    **$$##
    <text end>
'''

    messages.append(build_message(message, "user"))

    completion = gpt.chat_completion(messages)
    print(completion)