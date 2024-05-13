# %%
import os
import numpy as np
import copy
from openai import OpenAI
import cv2
# from google.colab.patches import cv2_imshow
from moviepy.editor import ImageSequenceClip
from PIL import Image
import torch


from io import StringIO
from contextlib import redirect_stdout

# imports for LMPs
import shapely
from time import sleep
from shapely.geometry import *
from shapely.affinity import *
from openai import RateLimitError, APIConnectionError

# imports for Franka Panda environment
from fyp_package.experiments import pick_and_place_env as franka_env
# import prompts for this experiment
from fyp_package.langsam_agent_playground import prompts

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# %%
class LMP:

    def __init__(self, name, cfg, fixed_vars, variable_vars):
        self._name = name
        self._cfg = cfg

        self.prompt_examples = self._cfg['prompt_examples']
        self.top_system_message = self._cfg['top_system_message']

        self._stop_tokens = list(self._cfg['stop'])

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''

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

            to_exec = code_str

            gvars = merge_dicts([self._fixed_vars, self._variable_vars])
            lvars = kwargs

            if not self._cfg['debug_mode']:
                stdout = exec_safe(to_exec, gvars, lvars)

            self.exec_hist += f'\n{to_exec}'

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

# %% [markdown]
# ## LMP Setup

# %% [markdown]
# ### LMP Wrapper

# %%
class LMP_wrapper():

  def __init__(self, env, cfg, render=False):
    self.env = env
    self._cfg = cfg
    self.object_names = list(self._cfg['env']['init_objs'])

    self._min_xy = np.array(self._cfg['env']['coords']['bottom_left'])
    self._max_xy = np.array(self._cfg['env']['coords']['top_right'])
    self._range_xy = self._max_xy - self._min_xy

    self._table_z = self._cfg['env']['coords']['table_z']
    self.render = render

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

  def get_robot_pos(self):
    # return robot end-effector xy position in robot base frame
    return self.env.get_ee_pos()

  def goto_pos(self, position_xy):
    # move the robot end-effector to the desired xy position while maintaining same z
    ee_xyz = self.env.get_ee_pos()
    position_xyz = np.concatenate([position_xy, ee_xyz[-1]])
    while np.linalg.norm(position_xyz - ee_xyz) > 0.01:
      self.env.movep(position_xyz)
      self.env.step_sim_and_render()
      ee_xyz = self.env.get_ee_pos()

  def follow_traj(self, traj):
    for pos in traj:
      self.goto_pos(pos)

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
    return ['top left corner', 'top right corner', 'bottom left corner', 'botom right corner'][corner_idx]

  def get_side_name(self, pos):
    side_positions = self.get_side_positions()
    side_idx = np.argmin(np.linalg.norm(side_positions - pos, axis=1))
    return ['top side', 'right side', 'bottom side', 'left side'][side_idx]
  
  def get_fixed_camera_image(self):
    return self.env.get_camera_image()
  
  def get_wrist_mounted_camera_image(self):
    wrist_pose, orientation = self.env.get


cfg_tabletop = {
  'lmps': {
    'tabletop_ui': {
      'top_system_message': [prompts.top_system_message],
      'prompt_examples': [prompts.get_camera_image_example],
      'model': 'gpt-3.5-turbo-0125',
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

lmp_tabletop_coords = {
        'top_left':     (-0.3 + 0.05, -0.2 - 0.05),
        'top_side':     (0,           -0.2 - 0.05),
        'top_right':    (0.3 - 0.05,  -0.2 - 0.05),
        'left_side':    (-0.3 + 0.05, -0.5,      ),
        'middle':       (0,           -0.5,      ),
        'right_side':   (0.3 - 0.05,  -0.5,      ),
        'bottom_left':  (-0.3 + 0.05, -0.8 + 0.05),
        'bottom_side':  (0,           -0.8 + 0.05),
        'bottom_right': (0.3 - 0.05,  -0.8 + 0.05),
        'table_z':       0.0,
      }


# %%
def setup_LMPs(env, cfg_tabletop):
  # LMP env wrapper
  cfg_tabletop = copy.deepcopy(cfg_tabletop)
  cfg_tabletop['env'] = dict()
  cfg_tabletop['env']['init_objs'] = list(env.obj_name_to_id.keys())
  cfg_tabletop['env']['coords'] = lmp_tabletop_coords
  LMP_env = LMP_wrapper(env, cfg_tabletop)
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
  variable_vars = {
      k: getattr(LMP_env, k)
      for k in [
          'denormalize_xy', 'get_obj_names',
          'get_corner_name', 'get_side_name',
          'get_camera_image',
      ]
  }

  # creating the LMP that deals w/ high-level language commands
  lmp_tabletop_ui = LMP(
      'tabletop_ui', cfg_tabletop['lmps']['tabletop_ui'], fixed_vars, variable_vars
  )

  return lmp_tabletop_ui


# %%
#@title Initialize Env { vertical-output: true }
num_blocks = 3 #@param {type:"slider", min:0, max:4, step:1}
num_bowls = 3 #@param {type:"slider", min:0, max:4, step:1}
high_resolution = False #@param {type:"boolean"}
high_frame_rate = False #@param {type:"boolean"}

# setup env and LMP
env = franka_env.PickPlaceEnv(render=True, high_res=high_resolution, high_frame_rate=high_frame_rate)
# block_list = np.random.choice(franka_env.ALL_BLOCKS, size=num_blocks, replace=False).tolist()
# bowl_list = np.random.choice(franka_env.ALL_BOWLS, size=num_bowls, replace=False).tolist()
block_list = franka_env.ALL_BLOCKS[:num_blocks]
bowl_list = franka_env.ALL_BOWLS[:num_bowls]
obj_list = block_list + bowl_list
_ = env.reset(obj_list)
lmp_tabletop_ui = setup_LMP(env, cfg_tabletop)

# # display env
# cv2.imshow("environment", cv2.cvtColor(env.get_camera_image(), cv2.COLOR_BGR2RGB))
# cv2.waitKey(1)

print('available objects:')
print(obj_list)

# %%

user_input = 'Please load the test_image and put a green box in the middle, save it as green_box_no_fill.jpg.'

env.cache_video = []

print('Running policy and recording video...')
# lmp_tabletop_ui(user_input, f'objects = {env.object_list}')
lmp_tabletop_ui(user_input)



# render video
if env.cache_video:
  rendered_clip = ImageSequenceClip(env.cache_video, fps=35 if high_frame_rate else 25)
  rendered_clip.write_gif("robot_clip.gif")