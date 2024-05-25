# %%
import os
import numpy as np
from openai import OpenAI
import cv2
import torch
from io import StringIO
from contextlib import redirect_stdout
from PIL import Image

# imports for LMPs
import shapely
from time import sleep
from shapely.geometry import *
from shapely.affinity import *
from openai import RateLimitError, APIConnectionError

from fyp_package import config, object_detection_utils, utils, model_client, environment
from fyp_package.experiments.no_ground_truth_prompts import *

client = OpenAI(api_key=utils.get_api_key())

# %%
cfg_vision_lmp = {
  'lmps': {
    'vision_assistant': {
      'top_system_message': [vision_top_system_message],
      'prompt_examples': [vision_get_camera_image_example],
      'model': 'gpt-4o',
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#', 'objects = ['],
      'maintain_session': True,
      'debug_mode': False,
      'include_context': True,
      'has_return': False,
      'return_val_name': 'ret_val',
    },
  }
}

class VisionLMP:

    def __init__(self, name, cfg, lmp_fgen, fixed_vars, variable_vars):
        self._name = name
        self._cfg = cfg

        self.prompt_examples = self._cfg['prompt_examples']
        self.top_system_message = self._cfg['top_system_message']

        self._stop_tokens = list(self._cfg['stop'])

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''
        self._lmp_fgen = lmp_fgen

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_initial_messages(self, query):
        if len(self._variable_vars) > 0:
            variable_vars_str = f"{', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_str = ''
        top_system_message = self.top_system_message.replace('{variable_vars}', variable_vars_str)

        messages = [self.build_message(top_system_message, 'system')]
        for prompt in self.prompt_examples:
            for i, message in enumerate(prompt):
                if i == 0:
                    messages.append(self.build_message(message, 'user'))
                elif i % 2 == 1:
                    messages.append(self.build_message(message, 'assistant'))
                else:
                    messages.append(self.build_message(message, 'system'))

        messages.append(self.build_message(query, 'user'))
        print('Initial messages:', messages)

        return messages
    
    def build_message(self, content, role):
       return {'role': role, 'content': content}

    def __call__(self, query, **kwargs):
        end = False
        messages = self.build_initial_messages(query)

        while not end:

            while True:
                try:
                    completion = client.chat.completions.create(
                        messages=messages,
                        stop=self._stop_tokens,
                        temperature=self._cfg['temperature'],
                        model=self._cfg['model'],
                        max_tokens=self._cfg['max_tokens']
                    ).choices[0].message.content.strip()
                    break
                except (RateLimitError, APIConnectionError) as e:
                    print(f'OpenAI API got err {e}')
                    print('Retrying after 10s.')
                    sleep(10)

            print(f'Completion: {completion}')
            messages.append(self.build_message(completion, 'assistant'))

            sections = completion.split('**')
            if len(sections) <= 1:
                print('Incorrect format, implement error correction')
                end = True
                break

            if sections[1] == 'END':
                end = True
                break

            code_str = sections[2]

            ## Function generation for subfunctions, if none are made this does nothing
            new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
            self._variable_vars.update(new_fs)

            gvars = merge_dicts([self._fixed_vars, self._variable_vars])
            lvars = kwargs

            if not self._cfg['debug_mode']:
                stdout = exec_safe(code_str, gvars, lvars)

            self.exec_hist += f'\n{code_str}'

            if self._cfg['maintain_session']:
                self._variable_vars.update(lvars)

            system_message = f'stdout: \n{stdout}'
            print(system_message)

            messages.append(self.build_message(system_message, 'system'))


def var_exists(name, all_vars):
    try:
        eval(name, all_vars)
    except:
        exists = False
    else:
        exists = True
    return exists


def merge_dicts(dicts):
    return {
        k : v
        for d in dicts
        for k, v in d.items()
    }


def exec_safe(code_str, gvars=None, lvars=None):
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])

    out = StringIO()
    with redirect_stdout(out):
        exec(code_str, custom_gvars, lvars)
    return out.getvalue()

# %%
def setup_vision_LMP(lmp_fgen, environment_vars):

  # creating APIs that the LMPs can interact with
  fixed_vars = {
      'np': np,
      'os': os,
      'cv2': cv2,
      'torch': torch,
      'PIL.Image': Image,
  }
  fixed_vars.update({
      name: eval(name)
      for name in shapely.geometry.__all__ + shapely.affinity.__all__
  })

  # creating the LMP that deals w/ vision
  lmp_vision_assistant = VisionLMP(
      'vision_assistant', cfg_vision_lmp['lmps']['vision_assistant'], lmp_fgen, fixed_vars, environment_vars
  )

  return lmp_vision_assistant