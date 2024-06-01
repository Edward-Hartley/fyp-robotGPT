import numpy as np
import copy
from io import StringIO
from contextlib import redirect_stdout

# imported just for agent to use
import shapely.geometry, shapely.affinity

from fyp_package import config, object_detection_utils, utils, model_client, environment, vision_agent, gpt_model
import fyp_package.prompts.robot_agent_prompts as prompts
from fyp_package.gpt_model import build_message, build_image_message

class RobotAgent:

    def __init__(self, name, cfg, fixed_vars, variable_vars, agent_env):
        self._name = name
        self._cfg = cfg
        self._env = agent_env

        self.prompt_examples = self._cfg['prompt_examples']
        self.top_system_message = self._cfg['top_system_message']
        self.final_system_message = self._cfg['final_system_message']
        self.check_completion_message = self._cfg['check_completion_message']

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self.gpt_model = gpt_model.GptModel(
            model=self._cfg['model'],
            stop=self._cfg['stop'],
            temperature=self._cfg['temperature'],
            max_tokens=self._cfg['max_tokens']
            )

    def build_initial_messages(self, query):
        functions_docs_str=''
        for function in self._variable_vars.keys():
            if function in prompts.function_docs:
                functions_docs_str += f"{function}:\n{prompts.function_docs[function]}\n\n"

        functions_advice_str = ''
        for function in self._variable_vars.keys():
            if function in prompts.functions_advice:
                functions_advice_str += f"{function}:\n{prompts.functions_advice[function]}\n\n"

        top_system_message = self.top_system_message.replace('{functions_docs}', functions_docs_str)
        top_system_message = top_system_message.replace('{packages}', str(list(self._fixed_vars.keys())))
        top_system_message = top_system_message.replace('{functions_advice}', functions_advice_str)
        if self.prompt_examples:
            top_system_message = top_system_message.replace('{few_shot_introduction}', prompts.few_shot_introduction)
        else:
            top_system_message = top_system_message.replace('{few_shot_introduction}', '')

        messages = [build_message(top_system_message, 'system')]
        for prompt in self.prompt_examples:
            for i, message in enumerate(prompt):
                if i == 0:
                    messages.append(build_message(message, 'user'))
                elif i % 2 == 1:
                    messages.append(build_message(message, 'assistant'))
                else:
                    messages.append(build_message(message, 'system'))

        if self.prompt_examples:
            messages.append(build_message(self.final_system_message, 'system'))

        messages.append(build_message(query, 'user'))

        print('Initial messages vision assistant')
        print(messages[-1])
        utils.log_completion(self._name, messages[-1], config.latest_generation_logs_path)

        # utils.print_openai_messages(messages[0])

        return messages

    def __call__(self, query, **kwargs):
        end = False
        messages = self.build_initial_messages(query)
        lvars = kwargs

        if self._cfg['include_gptv_context']:
            rgb, _ = self._env.get_images(save=False)
            utils.save_numpy_image(config.image_to_display_in_message_path, rgb)
            messages.append(build_message(prompts.gptv_injection_message, 'assistant'))
            utils.log_completion(self._name, prompts.gptv_injection_message, config.latest_generation_logs_path)
            messages.append(build_image_message(config.image_to_display_in_message_path))
            utils.log_viewed_image(config.image_to_display_in_message_path, config.viewed_image_logs_directory)

        while not end:

            completion = self.gpt_model.chat_completion(messages)

            print(f'Completion: {completion}')
            utils.log_completion(self._name, completion, config.latest_generation_logs_path)
            messages.append(build_message(completion, 'assistant'))

            sections = completion.split('##')
            if len(sections) <= 1:
                print('Incorrect format, implement error correction')
                end = True
                break


            gvars = merge_dicts([self._fixed_vars, self._variable_vars])

            if sections[1] == 'COMPLETE':
                end = self.confirm_complete(messages, query, lvars)
                if end:
                    break

            code_str = sections[2]

            stdout = exec_safe(code_str, gvars, lvars)

            self._variable_vars.update(lvars)

            system_message = f'stdout: \n{stdout}'
            print(system_message)
            utils.log_completion(self._name, system_message, config.latest_generation_logs_path)

            messages.append(build_message(system_message, 'system'))

            if "display_image(" in code_str:
                messages.append(build_image_message(config.image_to_display_in_message_path))
                utils.log_viewed_image(config.image_to_display_in_message_path, config.viewed_image_logs_directory)

    def confirm_complete(self, messages, query, lvars):
        rgb, _ = self._env.get_images(save=False)
        utils.save_numpy_image(config.image_to_display_in_message_path, rgb)
        # repeat user query and check for completion
        confirmation_message = self.check_completion_message.replace('{query}', query)
        messages.append(build_image_message(config.image_to_display_in_message_path, text=confirmation_message))
        utils.log_viewed_image(config.image_to_display_in_message_path, config.viewed_image_logs_directory)
        utils.log_completion(self._name, confirmation_message, config.latest_generation_logs_path)

        completion = self.gpt_model.chat_completion(messages)

        messages.append(build_message(completion, 'assistant'))
        utils.log_completion(self._name, completion, config.latest_generation_logs_path)
        
        if 'COMPLETE' in completion:
            return True
        return False


# class FGenAgent:


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


class EnvWrapper():

    def __init__(self, env: environment.Environment, cfg, render=False):
        self.env = env
        self._cfg = cfg
        self.object_names = list(self._cfg['env']['init_objs'])

        self._min_xy = np.array(self._cfg['env']['coords']['bottom left corner'][0:2])
        self._max_xy = np.array(self._cfg['env']['coords']['top right corner'][0:2])
        self._range_xy = self._max_xy - self._min_xy

        self._table_z = self._cfg['env']['coords']['table_z']
        self.render = render

        self.model_client = model_client.ModelClient()

    def get_obj_names(self):
        return self.object_names[::]

    def denormalize_xy(self, pos_normalized):
        return pos_normalized * self._range_xy + self._min_xy

    def get_corner_positions(self):
        unit_square = shapely.geometry.box(0, 0, 1, 1)
        normalized_corners = np.array(list(unit_square.exterior.coords))[:4]
        corners = np.array(([self.denormalize_xy(corner) for corner in normalized_corners]))
        return corners

    def get_side_positions(self):
        side_xs = np.array([0, 0.5, 0.5, 1])
        side_ys = np.array([0.5, 0, 1, 0.5])
        normalized_side_positions = np.c_[side_xs, side_ys]
        side_positions = np.array(([self.denormalize_xy(corner) for corner in normalized_side_positions]))
        return side_positions


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
    
    def put_first_on_second(self, pick_pos, place_pos, pick_angle=None):
        # put the source on top of target
        # place and place are x-y positions in robot base frame
        self.env.put_first_on_second(pick_pos, place_pos, pick_angle)

    def get_robot_pos(self):
        # return robot end-effector xyz position in robot base frame
        return self.env.get_ee_pose()[0]

    def move_robot(self, position_xyz, orientation_e=None):
        return self.env.move_robot(position_xyz, orientation_e, relative=False)
    
    def move_robot_relative(self, position_xyz, orientation_e=None):
        return self.env.move_robot(position_xyz, orientation_e, relative=True)
    
    def open_gripper(self):
        print("Gripper now open.")
        return self.env.open_gripper()
    
    def close_gripper(self):
        print("Gripper now closed.")
        return self.env.close_gripper()
    
    def detect_object(self, prompt, image, depth_array, camera_position=config.camera_position, camera_orientation_q=config.camera_orientation_q):
        masks, _, segmentation_texts = self.model_client.langsam_predict(image, prompt, save=True)

        return object_detection_utils.get_object_cube_from_segmentation(masks, segmentation_texts, image, depth_array, camera_position, camera_orientation_q, config.intrinsics), masks

    def get_images(self, save=True):
        return self.env.get_images(save=save)
    
    def display_image(self, image_array):
        image_array = np.uint8(np.array(image_array))
        # if mask, it'll be booleans, convert to image
        if image_array.dtype == bool:
            image_array = image_array * 255
        if image_array.ndim == 2:
            image_array = np.stack([image_array] * 3, axis=-1)

        print("Displaying image...")
        utils.save_numpy_image(config.image_to_display_in_message_path, image_array)

    def detect_grasp(self, mask, depth):
        depth_path = config.chosen_depth_image_path
        mask_path = config.chosen_segmentation_mask_path
        np.save(depth_path, depth)
        np.save(mask_path, mask)

        result = self.model_client.contact_graspnet_predict(depth_path=depth_path, rgb_path=None, mask_path=mask_path, save=True)
        if result is None:
            print("No grasp detected. Returned None, None.")
            return None
        grasp2cam_tf, _score, contact_point_cam = result

        grasp2base_tf = config.cam2base_tf @ grasp2cam_tf

        contact_point = config.cam2base_tf @ np.concatenate([contact_point_cam, [1]])[:3]
        # placeholder to remind this might be needed
        contact_point[2] -= 0.0

        grasp_position = utils.tf_trans(grasp2base_tf)
        grasp_orientation = utils.tf_rot(grasp2base_tf)
        grasp_orientation = utils.rot2quat(grasp_orientation)
        grasp_z_rot = -utils.quat2euler(grasp_orientation)[2]
        # wrap rot to [-pi/2, pi/2]
        if grasp_z_rot > np.pi/2:
            grasp_z_rot -= np.pi
        elif grasp_z_rot < -np.pi/2:
            grasp_z_rot += np.pi

        print("Detected grasp with contact point:", contact_point)
        print("Grasp rotation about z-axis:", grasp_z_rot)
        # print("Detected grasp with position:", grasp_position)
        # print("Detected grasp with orientation:", grasp_orientation)

        return contact_point, grasp_z_rot # , grasp_position, grasp_orientation


cfg_agent= {
    'robot_agent': {
        'top_system_message': prompts.top_system_message,
        'final_system_message': prompts.final_system_message,
        'check_completion_message': prompts.check_completion_message,
        'prompt_examples': [],
        'model': config.default_openai_model,
        'max_tokens': 512,
        'temperature': config.model_temperature,
        'stop': None,
        'include_gptv_context': True,
    },
}


def setup_agents(env: environment.Environment, cfg_agent):
    # agent env wrapper
    cfg_agent = copy.deepcopy(cfg_agent)
    cfg_agent['env'] = dict()
    cfg_agent['env']['init_objs'] = list(env.obj_list)
    cfg_agent['env']['coords'] = config.sim_corner_pos if config.simulation else config.real_corner_pos
    cfg_agent['env']['coords']['table_z'] = config.sim_table_z if config.simulation else config.real_table_z
    agent_env = EnvWrapper(env, cfg_agent)
    # creating APIs that the agents can interact with
    fixed_vars = {
        'np': np,
        'shapely.affinity': shapely.affinity,
        'shapely.geometry': shapely.geometry,
    }
    variable_vars = {
        k: getattr(agent_env, k)
        for k in [
            'denormalize_xy',
            'put_first_on_second', 'get_obj_names',
            'get_corner_name', 'get_side_name',
            'move_robot', 'move_robot_relative', 'open_gripper', 'close_gripper',
        ]
    }

    vision_variable_vars = {}
    vision_variable_vars['detect_object'] = agent_env.detect_object
    vision_variable_vars['get_images'] = agent_env.get_images
    vision_variable_vars['display_image'] = agent_env.display_image
    vision_variable_vars['detect_grasp'] = agent_env.detect_grasp

    # creating the vision agent for object detection
    vision_assistant = vision_agent.setup_vision_agent(environment_vars=vision_variable_vars)

    variable_vars['vision_assistant'] = vision_assistant

    # creating the function-generating agent
    # fgen_agent = FGenAgent()

    # creating the agent that deals w/ high-level language commands
    robot_agent = RobotAgent(
        'robot_agent', cfg_agent['robot_agent'], fixed_vars, variable_vars, agent_env
    )

    return robot_agent


# setup env and agent
if config.simulation:
  num_blocks = 3
  num_bowls = 3

  env = environment.SimulatedEnvironment(num_blocks, num_bowls)
else:
  env = environment.PhysicalEnvironment()
robot_agent = setup_agents(env, cfg_agent)

print('available objects:')
print(env.obj_list)

user_input = 'Put the red block in the red bowl'

# env.cache_video = []

print('Running policy and recording video...')
robot_agent(user_input)


# # render video
# if env.cache_video:
#   from moviepy.editor import ImageSequenceClip

#   rendered_clip = ImageSequenceClip(env.cache_video, fps=35 if high_frame_rate else 25)
#   rendered_clip.write_gif("robot_clip.gif")
