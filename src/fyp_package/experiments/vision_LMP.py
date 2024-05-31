# %%
from io import StringIO
from contextlib import redirect_stdout
from fyp_package import config, utils, gpt_model
from fyp_package.gpt_model import build_message, build_image_message
from fyp_package.experiments.CAP_vision_agent_prompts import *

# imports for LMPs
import numpy as np
import cv2
from PIL import Image

class VisionLMP:

    def __init__(self, name, cfg, fixed_vars, variable_vars):
        self._name = name
        self._cfg = cfg

        self.prompt_examples = self._cfg['prompt_examples']
        self.top_system_message = self._cfg['top_system_message']

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''

        self.gpt_model = gpt_model.GptModel(
            model=self._cfg['model'],
            stop=self._cfg['stop'],
            temperature=self._cfg['temperature'],
            max_tokens=self._cfg['max_tokens']
            )

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_initial_messages(self, query):
        functions_docs_str=''
        for function in self._variable_vars.keys():
            if function in vision_function_docs:
                functions_docs_str += f"{function}:\n{vision_function_docs[function]}\n\n"

        top_system_message = self.top_system_message.replace('{functions_docs}', functions_docs_str)
        top_system_message = top_system_message.replace('{packages}', str(list(self._fixed_vars.keys())[:-18]))

        messages = [build_message(top_system_message, 'system')]
        for prompt in self.prompt_examples:
            for i, message in enumerate(prompt):
                if i == 0:
                    messages.append(build_message(message, 'user'))
                elif i % 2 == 1:
                    messages.append(build_message(message, 'assistant'))
                else:
                    messages.append(build_message(message, 'system'))

        messages.append(build_message(vision_final_system_message, 'system'))
        messages.append(build_message(query, 'user'))
        print('Initial messages vision assistant')
        # utils.print_openai_messages(messages[0])

        return messages

    def __call__(self, query, **kwargs):
        end = False
        messages = self.build_initial_messages(query)
        lvars = kwargs

        while not end:

            completion = self.gpt_model.chat_completion(messages)

            print(f'Completion: {completion}')
            utils.log_completion(self._name, completion, config.latest_generation_logs_path)
            messages.append(build_message(completion, 'assistant'))

            sections = completion.split('**')
            if len(sections) <= 1:
                print('Incorrect format, implement error correction')
                end = True
                break

            code_str = sections[2]

            ## Function generation for subfunctions, if none are made this does nothing
            # new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
            # self._variable_vars.update(new_fs)

            gvars = merge_dicts([self._fixed_vars, self._variable_vars])

            if sections[1] == 'RET':
                end = True
                print("Returned value:", eval(code_str, gvars, lvars))
                return self.confirm_return(messages, eval(code_str, gvars, lvars), query, lvars)

            stdout = exec_safe(code_str, gvars, lvars)

            self.exec_hist += f'\n{code_str}'

            if self._cfg['maintain_session']:
                self._variable_vars.update(lvars)

            system_message = f'stdout: \n{stdout}'
            print(system_message)
            utils.log_completion(self._name, system_message, config.latest_generation_logs_path)

            messages.append(build_message(system_message, 'system'))

            if "display_image(" in code_str:
                messages.append(build_image_message(config.image_to_display_in_message_path, 'user'))
                utils.log_viewed_image(config.image_to_display_in_message_path, config.viewed_image_logs_directory)

    def confirm_return(self, messages, ret_val, query, lvars):
        confirmation_message = (
            vision_confirm_return
            .replace('{ret_val}', str(ret_val))
            .replace('{user_query}', query)
        )

        messages.append(build_message(confirmation_message, 'system'))
        utils.log_completion(self._name, confirmation_message, config.latest_generation_logs_path)
        completion = self.gpt_model.chat_completion(messages)

        utils.log_completion(self._name, completion, config.latest_generation_logs_path)
        messages.append(build_message(completion, 'assistant'))

        sections = completion.split('**')
        if len(sections) <= 1:
            print('Incorrect format, implement error correction')
            end = True
            return

        code_str = sections[2]

        ## Function generation for subfunctions, if none are made this does nothing
        # new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
        # self._variable_vars.update(new_fs)

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])

        if sections[1] == 'RET':
            return eval(code_str, gvars, lvars)


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
        try:
            exec(code_str, custom_gvars, lvars)
        except NameError as e:
            if e.args[0] == "name 'rgb' is not defined":
                exec("rgb, depth = get_images()", custom_gvars, lvars)
                exec(code_str, custom_gvars, lvars)
    return out.getvalue()

cfg_vision_lmp = {
  'lmps': {
    'vision_assistant': {
      'top_system_message': vision_top_system_message,
      'prompt_examples': [vision_detect_object_example, vision_detect_grasp_example],
      'model': config.default_openai_model,
      'max_tokens': 512,
      'temperature': config.model_temperature,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': None,
      'maintain_session': True,
      'include_context': True,
      'has_return': False,
      'return_val_name': 'ret_val',
    },
  }
}

# %%
def setup_vision_LMP(environment_vars={}):

  # creating APIs that the LMPs can interact with
  fixed_vars = {
      'np': np,
      'cv2': cv2,
      'PIL.Image': Image,
  }

  # creating the LMP that deals w/ vision
  lmp_vision_assistant = VisionLMP(
      'vision_assistant', cfg_vision_lmp['lmps']['vision_assistant'], fixed_vars, environment_vars
  )

  return lmp_vision_assistant

def dummy_get_images():
    print(input())
    return np.random.rand(480, 640, 3), np.random.rand(480, 640)

def dummy_detect_object(prompt, rgb, depth):
    print(input())
    fake_masks = [np.zeros((480, 480), dtype=bool) for _ in range(2)]
    fake_results = [
        {
            'position': [-0.001, -0.505, 0.049],
            'width': 0.587,
            'length': 0.608,
            'height': 0.063,
            'orientation': {'width': 1.568, 'length': -0.002}
        },
        {
            'position': [-0.166, -0.616, 0.048],
            'width': 0.042,
            'length': 0.045,
            'height': 0.048,
            'orientation': {'width': 0.03, 'length': -1.541}
        }
    ]
    return fake_results, fake_masks

def dummy_detect_grasp(mask, depth):
    print(input())
    return np.random.rand(3)

if __name__ == '__main__':
    environment_vars = {
        'get_images': dummy_get_images,
        'detect_object': dummy_detect_object,
        'detect_grasp': dummy_detect_grasp,
        }
    lmp_vision_assistant = setup_vision_LMP(None, environment_vars)

    lmp_vision_assistant('Return the [x, y, z] coordinates of where to grasp the red bowl')