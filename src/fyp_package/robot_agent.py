import traceback
import numpy as np
from io import StringIO
from contextlib import redirect_stdout

# imported just for agent to use
import shapely.geometry, shapely.affinity

from fyp_package import config, object_detection_utils, utils, model_client, environment, vision_agent, gpt_model, test_configurations, agent_logging
import fyp_package.prompts.robot_agent_prompts as prompts
import fyp_package.prompts.vision_agent_prompts as vision_prompts
from fyp_package.gpt_model import build_message, build_image_message

class RobotAgent:

    def __init__(self, name, cfg, fixed_vars, top_variable_vars, agent_env):
        self._name = name
        self._cfg = cfg
        self._env = agent_env

        self.prompt_examples = self._cfg['prompt_examples']
        self.top_system_message = self._cfg['top_system_message']
        self.final_system_message = self._cfg['final_system_message']
        self.check_completion_message = self._cfg['check_completion_message']

        self._vision_enabled = self._cfg['vision_enabled']

        self._fixed_vars = fixed_vars
        self._variable_vars = top_variable_vars

        self.gpt_model = gpt_model.GptModel(
            model=self._cfg['model'],
            stop=config.stop,
            temperature=config.model_temperature,
            max_tokens=config.max_tokens,
            name=self._name,
            )

        self.messages = []

    def add_message(self, message, generated=False):
        agent_logging.log_message(self._name, message, generated)
        self.messages.append(message)

    def build_initial_messages(self, query):
        self.messages = []
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
        top_system_message = top_system_message.replace('{table_bounds}', str(config.sim_corner_pos if config.simulation else config.real_corner_pos))

        self.add_message(build_message(top_system_message, 'system'))
        for prompt in self.prompt_examples:
            for i, message in enumerate(prompt):
                if i == 0:
                    self.add_message(build_message(message, 'user'))
                elif i % 2 == 1:
                    self.add_message(build_message(message, 'assistant'))
                else:
                    self.add_message(build_message(message, 'system'))

        if self.prompt_examples:
            self.add_message(build_message(self.final_system_message, 'system'))

        self.add_message(build_message(query, 'user'))

        print('Initial self.messages vision assistant')
        print(self.messages[-1])
        utils.log_completion(self._name, self.messages[-1], config.latest_generation_logs_path)

        # utils.print_openai_messages(self.messages)

        return self.messages
    
    def build_image_message_if_able(self, image_path, text=None):
        if self._vision_enabled:
            return build_image_message(image_path, text=text)
        else:
            return build_message('Vision is disabled, please carry out any other checks you can and then continue with the user\'s task.', 'system')

    def __call__(self, query, **kwargs):
        end = False
        self.build_initial_messages(query)
        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        gvars.update(kwargs)
        empty_fn = lambda *args, **kwargs: None
        gvars.update({'exec': empty_fn, 'eval': empty_fn})
        lvars=None

        attempts = 0

        if self._cfg['include_gptv_context']:
            rgb, _ = self._env.get_images(save=False)
            utils.save_numpy_image(config.image_to_display_in_message_path, rgb)
            self.add_message(build_message(prompts.gptv_injection_message, 'assistant'))
            utils.log_completion(self._name, prompts.gptv_injection_message, config.latest_generation_logs_path)
            self.add_message(self.build_image_message_if_able(config.image_to_display_in_message_path))
            utils.log_viewed_image(config.image_to_display_in_message_path, config.viewed_image_logs_directory)

        while not end:

            completion = self.gpt_model.chat_completion(self.messages)

            print(f'Completion: {completion}')
            utils.log_completion(self._name, completion, config.latest_generation_logs_path)
            self.add_message(build_message(completion, 'assistant'), generated=True)

            sections = completion.split('$$')
            if len(sections) <= 1:
                message = input('Please provide a correction for the tool use.\n')
                if message == '':
                    self.add_message(build_message(prompts.missing_tool_use_correction, 'system'))                
                else:
                    self.add_message(build_message(message, 'system'))
                continue


            if sections[1] == 'COMPLETE':
                attempts += 1
                if attempts >= 2:
                    break
                end, completion = self.confirm_complete(query)
                if end:
                    break
                else:
                    sections = completion.split('$$')
                    if len(sections) <= 1:
                        continue

            if sections[1] == 'CODE':

                code_str = sections[2]

                stdout = exec_safe(code_str, gvars, lvars)

                system_message = f'stdout: \n{stdout}'
                print(system_message)
                utils.log_completion(self._name, system_message, config.latest_generation_logs_path)

                self.add_message(build_message(system_message, 'system'))
                return

            if "$$VIEW_SCENE$$" in completion:
                rgb, _ = self._env.get_images(save=False)
                utils.save_numpy_image(config.image_to_display_in_message_path, rgb)
                self.add_message(self.build_image_message_if_able(config.image_to_display_in_message_path))
                utils.log_viewed_image(config.image_to_display_in_message_path, config.viewed_image_logs_directory)

    def confirm_complete(self, query):
        rgb, _ = self._env.get_images(save=False)
        utils.save_numpy_image(config.image_to_display_in_message_path, rgb)
        # repeat user query and check for completion
        confirmation_message = self.check_completion_message.replace('{query}', query)
        self.add_message(self.build_image_message_if_able(config.image_to_display_in_message_path, text=confirmation_message))
        utils.log_viewed_image(config.image_to_display_in_message_path, config.viewed_image_logs_directory)
        utils.log_completion(self._name, confirmation_message, config.latest_generation_logs_path)

        completion = self.gpt_model.chat_completion(self.messages)

        self.add_message(build_message(completion, 'assistant'), generated=True)
        utils.log_completion(self._name, completion, config.latest_generation_logs_path)
        
        if 'COMPLETE' in completion:
            return True, completion
        return False, completion


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


    out = StringIO()
    with redirect_stdout(out):
        exec(code_str, gvars, None)

    return out.getvalue()


class EnvWrapper():

    def __init__(self, env: environment.Environment, env_cfg):
        self.env = env
        self._cfg = env_cfg
        self.object_names = list(self._cfg['init_objs'])

        self._min_xy = np.array(self._cfg['coords']['bottom left corner'][0:2])
        self._max_xy = np.array(self._cfg['coords']['top right corner'][0:2])
        self._range_xy = self._max_xy - self._min_xy

        self._table_z = self._cfg['coords']['table_z']

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
        return self.env.put_first_on_second(pick_pos, place_pos, pick_angle)

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

        result = self.model_client.graspnet_predict(depth_path=depth_path, rgb_path=None, mask_path=mask_path, save=True)
        if result is None:
            print("No grasp detected. Returned None, None.")
            return None, None
        grasp2cam_tf, _score, contact_point_cam = result

        grasp2base_tf = config.cam2base_tf @ grasp2cam_tf

        contact_point = (config.cam2base_tf @ np.concatenate([contact_point_cam, [1]]))[:3]
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

def setup_agents(env: environment.Environment, cfg_agents, vision_assistant=True):
    # agent env wrapper
    env_cfg = dict()
    env_cfg['init_objs'] = list(env.obj_list)
    env_cfg['coords'] = config.sim_corner_pos if config.simulation else config.real_corner_pos
    env_cfg['coords']['table_z'] = config.sim_table_z if config.simulation else config.real_table_z
    agent_env = EnvWrapper(env, env_cfg)
    # creating APIs that the agents can interact with
    fixed_vars = {
        'np': np,
        'shapely.affinity': shapely.affinity,
        'shapely.geometry': shapely.geometry,
    }
    top_variable_vars = {
        k: getattr(agent_env, k)
        for k in cfg_agents['robot_agent']['functions']
    }

    if vision_assistant:
        vision_variable_vars = {
            k: getattr(agent_env, k)
            for k in cfg_agents['vision_assistant']['functions']
        }

        # creating the vision agent for object detection
        vision_assistant = vision_agent.setup_vision_agent(cfg_agents['vision_assistant'], environment_vars=vision_variable_vars)

        top_variable_vars['vision_assistant'] = vision_assistant

    # creating the function-generating agent
    # fgen_agent = FGenAgent()

    # creating the agent that deals w/ high-level language commands
    robot_agent = RobotAgent(
        'robot_agent', cfg_agents['robot_agent'], fixed_vars, top_variable_vars, agent_env
    )

    return robot_agent

_default_user_input = 'Move the red block into the red bowl.'

def run_agent(
        cfg_agents,
        user_input=_default_user_input,
        ):

    # setup env and agent
    if config.simulation:
        num_blocks = 3
        num_bowls = 3

        env = environment.SimulatedEnvironment(num_blocks, num_bowls)
    else:
        env = environment.PhysicalEnvironment()

    vision_assistant = 'vision_assistant' in cfg_agents

    robot_agent = setup_agents(env, cfg_agents, vision_assistant=vision_assistant)

    robot_agent(user_input)

if __name__ == '__main__':
#     agent_logging.setup_logging()
#     test_name = 'cap_top_and_vision'
#     cfg_agents = test_configurations.robotic_code_as_policy

#     user_query = 'Arrange the blocks in a line from the top left corner to the bottom right corner.'

#     full_configuration = {
#         'test_name': test_name,
#         'user_query': user_query,
#         'cfg_agents': cfg_agents,
#         'simulation': config.simulation,
#         'model_temperature': config.model_temperature,
#         'max_tokens': config.max_tokens,
#         'stop': config.stop,
#     }
    
#     agent_logging.log_configuration(full_configuration, test_name)

#     try:
#         run_agent(cfg_agents, user_query)
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         print(f"error traceback: {traceback.format_exc()}")
#         agent_logging.log_event('Error occurred', f"Error: {e}")
#     finally:
#         agent_logging.print_run_id()
#         agent_logging.log_final_notes()

    agent_logging.setup_logging()
    test_name = 'cap_vision_agent'
    cfg_agents = test_configurations.robotic_code_as_policy

    user_query = 'Varied user query to test vision agent capabilities.'

    full_configuration = {
        'test_name': test_name,
        'user_query': user_query,
        'cfg_agents': cfg_agents,
        'simulation': config.simulation,
        'model_temperature': config.model_temperature,
        'max_tokens': config.max_tokens,
        'stop': config.stop,
    }
    
    agent_logging.log_configuration(full_configuration, test_name)

    env = environment.PhysicalEnvironment()

    env_cfg = dict()
    env_cfg['init_objs'] = list(env.obj_list)
    env_cfg['coords'] = config.sim_corner_pos if config.simulation else config.real_corner_pos
    env_cfg['coords']['table_z'] = config.sim_table_z if config.simulation else config.real_table_z
    agent_env = EnvWrapper(env, env_cfg)

    vision_variable_vars = {
        k: getattr(agent_env, k)
        for k in cfg_agents['vision_assistant']['functions']
    }
    vision_agent.DEBUG = True
    # creating the vision agent for object detection
    vision_assistant = vision_agent.setup_vision_agent(cfg_agents['vision_assistant'], environment_vars=vision_variable_vars)


    while user_query != 'exit':


        user_query = input("Please enter a user query: ")
        if user_query == 'exit':
            break
        agent_logging.log_event('User query', user_query)

        rgb, depth = env.get_images(save=False)
        utils.save_numpy_image(config.image_to_display_in_message_path, rgb)

        try:
            print(vision_assistant(user_query))
            # robot_pos = input("Please enter the robot position: ")
            # env.move_robot(eval(robot_pos))
        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"error traceback: {traceback.format_exc()}")
            agent_logging.log_event('Error occurred', f"Error: {e}")
        finally:
            agent_logging.print_run_id()
            agent_logging.log_final_notes()
        
