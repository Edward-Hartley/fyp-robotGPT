# %%
import os
import numpy as np
import copy
from openai import OpenAI
# import cv2
# from google.colab.patches import cv2_imshow
from PIL import Image

# imports for LMPs
import shapely
import ast
import astunparse
from time import sleep
from shapely.geometry import *
from shapely.affinity import *
from openai import RateLimitError, APIConnectionError
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

from fyp_package import config, object_detection_utils, utils, model_client, environment
from fyp_package.experiments.CAP_vision_agent_prompts import *
from fyp_package.experiments import vision_LMP

client = OpenAI(api_key=utils.get_api_key())

# %%
class LMP:

    def __init__(self, name, cfg, lmp_fgen, fixed_vars, variable_vars):
        self._name = name
        self._cfg = cfg

        self._base_prompt = self._cfg['prompt_text']

        self._stop_tokens = list(self._cfg['stop'])

        self._lmp_fgen = lmp_fgen

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query, context=''):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session']:
            prompt += f'\n{self.exec_hist}'

        if context != '':
            prompt += f'\n{context}'

        use_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{use_query}'

        return prompt, use_query

    def __call__(self, query, context='', **kwargs):
        prompt, use_query = self.build_prompt(query, context=context)

        while True:
            try:
                code_str = client.completions.create(
                    prompt=prompt,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['model'],
                    max_tokens=self._cfg['max_tokens']
                ).choices[0].text.strip()
#                 code_str = '''
# objects = ['espresso cup', 'bowl']
# # Put the espresso cup in the bowl.
# matches = {'espresso cup': 'bowl'}
# say('Got it - putting the espresso cup in the bowl')
# for first, second in matches.items():
#   put_first_on_second(get_obj_pos(first), get_obj_pos(second))'''
#                 code_str = '''
# objects = ['blue block', 'red block', 'green block', 'blue bowl', 'red bowl', 'green bowl']
# # Put the red block in the corresponding bowl.
# matches = {'red block': 'red bowl'}
# say('Got it - putting the red block in the red bowl')
# for first, second in matches.items():
#   put_first_on_second(get_obj_pos(first), get_obj_pos(second))'''
#                 code_str = '''
# objects = ['espresso cup', 'bowl']
# # Put the espresso cup in the bowl.
# matches = {'espresso cup': 'bowl'}
# say('Got it - putting the espresso cup in the bowl')
# for first, second in matches.items():
#     first_object_pos = vision_assistant(f"Please return the [x, y, z] position of the {first}.")
#     second_object_pos = vision_assistant(f"Please return the [x, y, z] position of the {second}.")
#     put_first_on_second(first_object_pos, second_object_pos)
#                 '''
                break
            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)

        if self._cfg['include_context'] and context != '':
            to_exec = f'{context}\n{code_str}'
            to_log = f'{context}\n{use_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{use_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())
        print(f'LMP {self._name} exec:\n\n{to_log_pretty}\n')
        utils.log_completion(self._name, to_log, config.latest_generation_logs_path)

        new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
        self._variable_vars.update(new_fs)

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        if not self._cfg['debug_mode']:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_exec}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            return lvars[self._cfg['return_val_name']]


class LMPFGen:

    def __init__(self, cfg, fixed_vars, variable_vars):
        self._cfg = cfg

        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self._base_prompt = self._cfg['prompt_text']

    def create_f_from_sig(self, f_name, f_sig, other_vars=None, fix_bugs=False, return_src=False):
        print(f'Creating function: {f_sig}')

        use_query = f'{self._cfg["query_prefix"]}{f_sig}{self._cfg["query_suffix"]}'
        prompt = f'{self._base_prompt}\n{use_query}'

        while True:
            try:
                f_src = client.completions.create(
                    prompt=prompt,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['model'],
                    max_tokens=self._cfg['max_tokens']
                ).choices[0].text.strip()
                break
            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)

        if fix_bugs:
            f_src = client.edits.create(
                model='code-davinci-edit-001',
                input='# ' + f_src,
                temperature=0,
                instruction='Fix the bug if there is one. Improve readability. Keep same inputs and outputs. Only small changes. No comments.',
            ).choices[0].text.strip()

        if other_vars is None:
            other_vars = {}
        gvars = merge_dicts([self._fixed_vars, self._variable_vars, other_vars])
        lvars = {}

        exec_safe(f_src, gvars, lvars)

        f = lvars[f_name]

        to_print = highlight(f'{use_query}\n{f_src}', PythonLexer(), TerminalFormatter())
        print(f'LMP FGEN created:\n\n{to_print}\n')

        if return_src:
            return f, f_src
        return f

    def create_new_fs_from_code(self, code_str, other_vars=None, fix_bugs=False, return_src=False):
        fs, f_assigns = {}, {}
        f_parser = FunctionParser(fs, f_assigns)
        f_parser.visit(ast.parse(code_str))
        for f_name, f_assign in f_assigns.items():
            if f_name in fs:
                fs[f_name] = f_assign

        if other_vars is None:
            other_vars = {}

        new_fs = {}
        srcs = {}
        for f_name, f_sig in fs.items():
            all_vars = merge_dicts([self._fixed_vars, self._variable_vars, new_fs, other_vars])
            if not var_exists(f_name, all_vars):
                f, f_src = self.create_f_from_sig(f_name, f_sig, new_fs, fix_bugs=fix_bugs, return_src=True)

                # recursively define child_fs in the function body if needed
                f_def_body = astunparse.unparse(ast.parse(f_src).body[0].body)
                child_fs, child_f_srcs = self.create_new_fs_from_code(
                    f_def_body, other_vars=all_vars, fix_bugs=fix_bugs, return_src=True
                )

                if len(child_fs) > 0:
                    new_fs.update(child_fs)
                    srcs.update(child_f_srcs)

                    # redefine parent f so newly created child_fs are in scope
                    gvars = merge_dicts([self._fixed_vars, self._variable_vars, new_fs, other_vars])
                    lvars = {}

                    exec_safe(f_src, gvars, lvars)

                    f = lvars[f_name]

                new_fs[f_name], srcs[f_name] = f, f_src

        if return_src:
            return new_fs, srcs
        return new_fs


class FunctionParser(ast.NodeTransformer):

    def __init__(self, fs, f_assigns):
      super().__init__()
      self._fs = fs
      self._f_assigns = f_assigns

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.func).strip()
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.value.func).strip()
            self._f_assigns[f_name] = assign_str
        return node


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
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    exec(code_str, custom_gvars, lvars)

# %% [markdown]
# ## LMP Setup

# %% [markdown]
# ### LMP Wrapper

# %%
class LMP_wrapper():

    def __init__(self, env: environment.Environment, cfg, render=False):
        self.env = env
        self._cfg = cfg
        self.object_names = list(self._cfg['env']['init_objs'])

        self._min_xy = np.array(self._cfg['env']['coords']['bottom left corner'][0:2])
        self._max_xy = np.array(self._cfg['env']['coords']['top right corner'][0:2])
        self._range_xy = self._max_xy - self._min_xy

        self._table_z = self._cfg['env']['coords']['table_z']
        self.render = render

        self.langsam_model = model_client.ModelClient()

    def get_obj_names(self):
        return self.object_names[::]

    def denormalize_xy(self, pos_normalized):
        return pos_normalized * self._range_xy + self._min_xy

    def get_corner_positions(self):
        unit_square = box(0, 0, 1, 1)
        normalized_corners = np.array(list(unit_square.exterior.coords))[:4]
        corners = np.array(([self.denormalize_xy(corner) for corner in normalized_corners]))
        return corners

    def get_side_positions(self):
        side_xs = np.array([0, 0.5, 0.5, 1])
        side_ys = np.array([0.5, 0, 1, 0.5])
        normalized_side_positions = np.c_[side_xs, side_ys]
        side_positions = np.array(([self.denormalize_xy(corner) for corner in normalized_side_positions]))
        return side_positions

    def put_first_on_second(self, pick_pos, place_pos):
        # put the source on top of target
        # place and place are x-y positions in robot base frame
        self.env.put_first_on_second(pick_pos, place_pos)

    def get_robot_pos(self):
        # return robot end-effector xyz position in robot base frame
        return self.env.get_ee_pose()[0]

    def move_robot(self, position_xyz):
        return self.env.move_robot(position_xyz, relative=False)

    def follow_traj(self, traj):
        for pos in traj:
            self.move_robot(pos)

    def get_corner_positions(self):
        normalized_corners = np.array([
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 0]
        ])
        return np.array(([self.denormalize_xy(corner) for corner in normalized_corners]))

    def get_side_positions(self):
        normalized_sides = np.array([
            [0.5, 1],
            [1, 0.5],
            [0.5, 0],
            [0, 0.5]
        ])
        return np.array(([self.denormalize_xy(side) for side in normalized_sides]))

    def get_corner_name(self, pos):
        corner_positions = self.get_corner_positions()
        corner_idx = np.argmin(np.linalg.norm(corner_positions - pos, axis=1))
        return ['top left corner', 'top right corner', 'bottom left corner', 'bottom right corner'][corner_idx]

    def get_side_name(self, pos):
        side_positions = self.get_side_positions()
        side_idx = np.argmin(np.linalg.norm(side_positions - pos, axis=1))
        return ['top side', 'right side', 'bottom side', 'left side'][side_idx]
    
    def detect_object(self, prompt, image, depth_array, camera_position=config.camera_position, camera_orientation_q=config.camera_orientation_q):
#         print(f'''
# Detection 1
# Position of {prompt}: [-0.001, -0.505, 0.049]
# Dimensions:
# Width: 0.587
# Length: 0.608
# Height: 0.063
# Orientation along shorter side (width): 1.568
# Orientation along longer side (length): -0.002  

# Detection 2
# Position of {prompt}: [-0.166, -0.616, 0.048]
# Dimensions:
# Width: 0.042
# Length: 0.045
# Height: 0.048
# Orientation along shorter side (width): 0.03
# Orientation along longer side (length): -1.541

# Total number of detections made: 2
# ''')
#         fake_masks = [np.zeros_like(image[:, :, 0], dtype=bool) for _ in range(2)]
#         fake_results = [
#             {
#                 'position': [-0.001, -0.505, 0.049],
#                 'width': 0.587,
#                 'length': 0.608,
#                 'height': 0.063,
#                 'orientation': {'width': 1.568, 'length': -0.002}
#             },
#             {
#                 'position': [-0.166, -0.616, 0.048],
#                 'width': 0.042,
#                 'length': 0.045,
#                 'height': 0.048,
#                 'orientation': {'width': 0.03, 'length': -1.541}
#             }
#         ]
#         return fake_results, fake_masks


        masks, _, segmentation_texts = self.langsam_model.langsam_predict(image, prompt, save=True)

        return object_detection_utils.get_object_cube_from_segmentation(masks, segmentation_texts, image, depth_array, camera_position, camera_orientation_q, config.intrinsics), masks

    def get_images(self):
        return self.env.get_images()
    
    def display_image(self, array_or_image):
            # Save in image_to_display_in_message_path
        if isinstance(array_or_image, np.ndarray):
            # if mask, it'll be booleans, convert to image
            if array_or_image.dtype == bool:
                array_or_image = array_or_image.astype(np.uint8) * 255
            if array_or_image.ndim == 2:
                array_or_image = np.stack([array_or_image] * 3, axis=-1)
            image = Image.fromarray(array_or_image, mode='RGB')
        else:
            image = array_or_image
        image.save(config.image_to_display_in_message_path)


# %% [markdown]
# ### LMP Config

# %%
cfg_tabletop = {
  'lmps': {
    'tabletop_ui': {
      'prompt_text': prompt_tabletop_ui,
      'model': config.completions_compatible_openai_model,
      'max_tokens': 512,
      'temperature': config.model_temperature,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#', 'objects = ['],
      'maintain_session': True,
      'debug_mode': False,
      'include_context': True,
      'has_return': False,
      'return_val_name': 'ret_val',
    },
    'parse_obj_name': {
      'prompt_text': prompt_parse_obj_name,
      'model': config.completions_compatible_openai_model,
      'max_tokens': 512,
      'temperature': config.model_temperature,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#', 'objects = ['],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
      'has_return': True,
      'return_val_name': 'ret_val',
    },
    'parse_position': {
      'prompt_text': prompt_parse_position,
      'model': config.completions_compatible_openai_model,
      'max_tokens': 512,
      'temperature': config.model_temperature,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#'],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
      'has_return': True,
      'return_val_name': 'ret_val',
    },
    'parse_question': {
      'prompt_text': prompt_parse_question,
      'model': config.completions_compatible_openai_model,
      'max_tokens': 512,
      'temperature': config.model_temperature,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#', 'objects = ['],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
      'has_return': True,
      'return_val_name': 'ret_val',
    },
    'transform_shape_pts': {
      'prompt_text': prompt_transform_shape_pts,
      'model': config.completions_compatible_openai_model,
      'max_tokens': 512,
      'temperature': config.model_temperature,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#'],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
      'has_return': True,
      'return_val_name': 'new_shape_pts',
    },
    'fgen': {
      'prompt_text': prompt_fgen,
      'model': config.completions_compatible_openai_model,
      'max_tokens': 512,
      'temperature': config.model_temperature,
      'query_prefix': '# define function: ',
      'query_suffix': '.',
      'stop': ['# define', '# example'],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
    }
  }
}

# %% [markdown]
# ### LMP Utils

# %%
def setup_LMP(env: environment.Environment, cfg_tabletop):
  # LMP env wrapper
  cfg_tabletop = copy.deepcopy(cfg_tabletop)
  cfg_tabletop['env'] = dict()
  cfg_tabletop['env']['init_objs'] = list(env.obj_list)
  cfg_tabletop['env']['coords'] = config.sim_corner_pos if config.simulation else config.real_corner_pos
  cfg_tabletop['env']['coords']['table_z'] = config.sim_table_z if config.simulation else config.real_table_z
  LMP_env = LMP_wrapper(env, cfg_tabletop)
  # creating APIs that the LMPs can interact with
  fixed_vars = {
      'np': np
  }
  fixed_vars.update({
      name: eval(name)
      for name in shapely.geometry.__all__ + shapely.affinity.__all__
  })
  variable_vars = {
      k: getattr(LMP_env, k)
      for k in [
          'denormalize_xy',
          'put_first_on_second', 'get_obj_names',
          'get_corner_name', 'get_side_name',

          # 'get_obj_pos', 'is_obj_visible'
      ]
  }
  variable_vars['say'] = lambda msg: print(f'robot says: {msg}')

  vision_variable_vars = {}
  vision_variable_vars['detect_object'] = LMP_env.detect_object
  vision_variable_vars['get_images'] = LMP_env.get_images
#   vision_variable_vars['display_image'] = LMP_env.display_image # not yet implemented

  # creating the vision LMP for object detection
  lmp_vision = vision_LMP.setup_vision_LMP(lmp_fgen=None, environment_vars=vision_variable_vars)

  variable_vars['vision_assistant'] = lmp_vision

  # creating the function-generating LMP
  lmp_fgen = LMPFGen(cfg_tabletop['lmps']['fgen'], fixed_vars, merge_dicts([variable_vars, vision_variable_vars]))

  # creating other low-level LMPs
  variable_vars.update({
      k: LMP(k, cfg_tabletop['lmps'][k], lmp_fgen, fixed_vars, variable_vars)
      for k in ['parse_obj_name', 'parse_position', 'parse_question', 'transform_shape_pts']
  })


  # creating the LMP that deals w/ high-level language commands
  lmp_tabletop_ui = LMP(
      'tabletop_ui', cfg_tabletop['lmps']['tabletop_ui'], lmp_fgen, fixed_vars, variable_vars
  )

  return lmp_tabletop_ui


# setup env and LMP
if config.simulation:
  num_blocks = 3
  num_bowls = 3

  env = environment.SimulatedEnvironment(num_blocks, num_bowls)
else:
  env = environment.PhysicalEnvironment()
lmp_tabletop_ui = setup_LMP(env, cfg_tabletop)

print('available objects:')
print(env.obj_list)

user_input = 'Put the red block in the corresponding bowl'

# env.cache_video = []

print('Running policy and recording video...')
lmp_tabletop_ui(user_input, f'objects = {env.obj_list}')



# # render video
# if env.cache_video:
#   from moviepy.editor import ImageSequenceClip

#   rendered_clip = ImageSequenceClip(env.cache_video, fps=35 if high_frame_rate else 25)
#   rendered_clip.write_gif("robot_clip.gif")
