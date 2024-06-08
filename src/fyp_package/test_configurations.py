from fyp_package.prompts import robot_agent_prompts as r_prompts, vision_agent_prompts as v_prompts
from fyp_package import config
import copy

all_features = {
        'robot_agent': {
            'top_system_message': r_prompts.top_system_message,
            'final_system_message': r_prompts.final_system_message,
            'check_completion_message': r_prompts.check_completion_message,
            'prompt_examples': [r_prompts.all_modules_example, r_prompts.grasping_example_snippet],
            'model': config.default_openai_model,
            'vision_enabled': True,
            'include_gptv_context': True,
            'functions' : [
                'denormalize_xy',
                'put_first_on_second', 'get_obj_names',
                'get_corner_name', 'get_side_name',
                'move_robot', 'move_robot_relative', 'open_gripper', 'close_gripper',
            ],
        },

        'vision_assistant': {
            'top_system_message': v_prompts.vision_top_system_message,
            'prompt_examples': [
                v_prompts.vision_detect_object_example,
                v_prompts.vision_detect_grasp_example
                ],
            'model': config.default_openai_model,
            'include_gptv_context': True,
            'functions' : [
                'detect_object', 'get_images', 'display_image', 'detect_grasp'
            ],
        },
    }

no_gptv_context = copy.deepcopy(all_features)
no_gptv_context['robot_agent']['include_gptv_context'] = False
no_gptv_context['vision_assistant']['include_gptv_context'] = False

remove_modules = {}
remove_modules['grasp'] = copy.deepcopy(all_features)
remove_modules['grasp']['vision_assistant']['functions'].remove('detect_grasp')
remove_modules['grasp']['vision_assistant']['prompt_examples'] = [
    v_prompts.vision_detect_object_example,
    v_prompts.vision_display_image_no_grasp_example,
]
remove_modules['grasp']['robot_agent']['prompt_examples'] = [
    r_prompts.all_modules_example,
    # r_prompts.grasping_example_snippet
]

remove_modules['display'] = copy.deepcopy(all_features)
remove_modules['display']['vision_assistant']['functions'].remove('display_image')
remove_modules['display']['vision_assistant']['include_gptv_context'] = False
remove_modules['display']['vision_assistant']['prompt_examples'] = [
    v_prompts.vision_generic_example, # if never using detect grasp, then add new example
    v_prompts.vision_detect_object_example,
]

remove_modules['grasp_and_display'] = copy.deepcopy(all_features)
remove_modules['grasp_and_display']['vision_assistant']['functions'].remove('display_image')
remove_modules['grasp_and_display']['vision_assistant']['functions'].remove('detect_grasp')
remove_modules['grasp_and_display']['vision_assistant']['include_gptv_context'] = False
remove_modules['grasp_and_display']['vision_assistant']['prompt_examples'] = [
    v_prompts.vision_generic_example, 
    v_prompts.vision_detect_object_example,
]
remove_modules['grasp_and_display']['robot_agent']['prompt_examples'] = [
    r_prompts.all_modules_example,
]

remove_modules['object_and_grasp'] = copy.deepcopy(all_features)
remove_modules['object_and_grasp']['vision_assistant']['functions'].remove('detect_object')
remove_modules['object_and_grasp']['vision_assistant']['functions'].remove('detect_grasp')
remove_modules['object_and_grasp']['vision_assistant']['include_gptv_context'] = True
remove_modules['object_and_grasp']['prompt_examples'] = [
    v_prompts.vision_generic_example,
    v_prompts.vision_get_images_example,
]
remove_modules['object_and_grasp']['robot_agent']['prompt_examples'] = [
    r_prompts.all_modules_example,
]

remove_modules['trajectory_control'] = copy.deepcopy(all_features)
for function in ['move_robot', 'move_robot_relative', 'open_gripper', 'close_gripper']:
    remove_modules['trajectory_control']['robot_agent']['functions'].remove(function)
remove_modules['trajectory_control']['robot_agent']['prompt_examples'] = [
    r_prompts.no_fine_control_example,
]

remove_modules['no_pick_and_place'] = copy.deepcopy(all_features)
remove_modules['no_pick_and_place']['robot_agent']['functions'].remove('put_first_on_second')
remove_modules['no_pick_and_place']['robot_agent']['prompt_examples'] = [
    r_prompts.no_pick_and_place_example,
    r_prompts.grasping_example_snippet
]

remove_modules['no_top_level_vision'] = copy.deepcopy(all_features)
remove_modules['no_top_level_vision']['robot_agent']['vision_enabled'] = False

remove_fewshot = {}
remove_fewshot['top_only'] = copy.deepcopy(all_features)
remove_fewshot['top_only']['robot_agent']['prompt_examples'] = [r_prompts.super_simple_examples]

remove_fewshot['top_and_vision'] = copy.deepcopy(all_features)
remove_fewshot['top_and_vision']['robot_agent']['prompt_examples'] = [r_prompts.super_simple_examples]
remove_fewshot['top_and_vision']['vision_assistant']['prompt_examples'] = [v_prompts.vision_generic_example]

minimal_prompts = {}
minimal_prompts['top_and_vision'] = copy.deepcopy(all_features)
minimal_prompts['top_and_vision']['robot_agent']['top_system_message'] = r_prompts.top_system_message_minimal
minimal_prompts['top_and_vision']['vision_assistant']['top_system_message'] = v_prompts.vision_top_system_message_minimal
# don't need to change prompt examples at the moment

robotic_code_as_policy = copy.deepcopy(all_features)
robotic_code_as_policy['robot_agent']['top_system_message'] = r_prompts.cap_top_system_message
robotic_code_as_policy['robot_agent']['prompt_examples'] = [r_prompts.cap_format_example]
robotic_code_as_policy['vision_assistant']['top_system_message'] = v_prompts.vision_cap_top_system_message
robotic_code_as_policy['vision_assistant']['prompt_examples'] = [v_prompts.vision_cap_format_example]

