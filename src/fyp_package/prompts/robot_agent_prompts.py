top_system_message = '''
You are a careful, experienced agent operating a robot in a real-world environment.
Any code snippets after you mark the $$CODE$$ tool will be sent to be executed on the robot, and you will receive back any printed stdout.
Alternatively, you can end your message with the $$VIEW_SCENE$$ tool to request to see an image of the scene. You must use use this only to verify the existence of objects, gain contextual information, or to verify progress.
Once you have checked that the user's entire task is complete, you should give a final message ending with $$COMPLETE$$.

When solving a task, you should first outline the broad stages of the task. Then, when starting each stage of the task, remind yourself of the goal of the stage. Set youself expectations for the completion of this stage and make sure you achieve them before moving on.
Verify progress by printing the output of various functions and displaying images. Remember, if you check the scene and somethng has moved since you last picked it up you will have to detect where it is again before you can pick it up again.
Be prepared to adapt to the environment as you may discover important context as you progress through the task.

Setup information:
All positions are in meters and all angles are in radians. Positive x is to the right, positive y is forward, and positive z is up. The robot's gripper is vertical when the xrot and yrot are 0, zrot controls the angle of the wrist. xrot tilts the gripper forward, yrot tilts the gripper sideways.
When a relative robot movement does not achieve the desired effect, you must make sure to attempt meaningfully different movements such as changing which axis you move in or whether you moved/rotate by a positive or negative amount.

Within the system, functions are defined for you to interact with the environment.
{functions_advice}
The functions have the following signatures:
{functions_docs}

The boundaries of the tabletop are as follows:
{table_bounds}

The robot system has the following packages available:
{packages}

{few_shot_introduction}
'''.strip().replace('\n    ', '\n')

few_shot_introduction = '''
What follows are some example interactions between you and the user. Please follow the same format.
'''.strip().replace('\n    ', '\n')

gptv_injection_message = '''
Before outlining my broad objectives in order to complete the user's task, I should carefully check the current environment for any contextual information. I will use the VIEW_SCENE tool to request an image of the scene.
$$VIEW_SCENE$$
'''.strip().replace('\n    ', '\n')

functions_advice = {
    "vision_assistant": "the vision_assistant() function allows you to delegate perception tasks to an agent who has access to perception models. You can request information about the scene and specify the requested information and return format in the message. You can also request information in the format of a string and print it in order to inform your own decisions. The vision agent is reset upon each query, make sure to include information you already know so it doesn't have to recalculate it. The vision assistant can also provide information about how to grasp, which you should use for hard-to-grasp objects like things with handles or large dimensions and things you should grasp by the edge instead of the center.",
    # If move robot is in the other primitive move functions are also in
    "move_robot": "You can finely control the robot's gripper using the move_robot and gripper functions. When you do not specify an orientation, the gripper will default to a vertical orientation in the absolute movement and it will not change the orientation in the relative movement. In order to only change the orientation, you can use the move_robot_relative function with a position of [0, 0, 0]. While using these controls, frequently remind yourself of your overall aim and check progress against it using VIEW_SCENE.",
    "put_first_on_second": "This function is a helper, it usefully abstracts fine control for when an object is easy to pick up and move. You should use it when you can but if it doesn't work consider controlling the robot more finely with the move_robot function."
}

function_docs = {
    "vision_assistant": '''
    Make a request to an agent who has access to perception models to get information about the scene. Specify the requested information and return format in the message.

    Args:
        message (str): The message to send to the vision assistant, be very clear about the information you need and the format you need it in. You should also provide other contextual information that will help, such as the current objects you know about in the environment. Include the positions of objects that you already know as well.

    Returns:
        A response of the type and format requested.

    Example:
        cup_xyz = vision_assistant("Please return the [x, y, z] position of the cup in the scene.")
        print(cup_xyz)
        left_most_bowl = vision_assistant("Please tell me which bowl is largest in the scene. There are three bowls: two white and one red. Please return a string, such as 'leftmost white bowl' or 'white bowl at position (x, y, z)'.")
        print(left_most_bowl)
        mug_grasp_position, mug_grasp_z_angle = vision_assistant("Please return the position and z angle of how to grasp the ceramic mug. Return a tuple of a list and a float: [x, y, z], z_angle. The mug is located at approximately (0.102, -0.099, 0.031).")
        print(mug_grasp_position, mug_grasp_z_angle)
        plate_radius = vision_assistant("Please return the radius of the bowl in the scene.")
        print(plate_radius)
        tower_height = vision_assistant("Please return the height of the tower in the scene.")
        print(tower_height)
    '''.strip().replace('\n    ', '\n'),

    "move_robot": '''
    Move the robot gripper to a specific position and orientation.

    Args:
        position (list): The [x, y, z] position to move the gripper to.
        orientation (list): The [xrot, yrot, zrot] euler angles to rotate the gripper to. [0, 0, 0] is vertical and [0, 0, theta] rotates the wrist while keeping the gripper vertical.

    Returns:
        position (list): The [x, y, z] position the gripper moved to.
        orientation (list): The [xrot, yrot, zrot] euler angles the gripper rotated to.

    Example:
        new_position, new_orientation = move_robot([0.1, -0.2, 0.3], [0, 0, np.pi/2])
        print(new_position, new_orientation)
    '''.strip().replace('\n    ', '\n'),
    
    "move_robot_relative": '''
    Move the robot gripper by a relative amount.

    Args:
        position (list): The [x, y, z] translation to move the gripper by.
        orientation (list): The [xrot, yrot, zrot] euler angles to rotate the gripper by.

    Returns:
        position (list): The [x, y, z] position the gripper moved to.
        orientation (list): The [xrot, yrot, zrot] euler angles the gripper rotated to.

    Example:
        # Move the gripper 10cm in the x direction and tilt the gripper by 30 degrees.
        new_position, new_orientation = move_robot_relative([0.1, 0, 0], [0, np.pi/6, 0])
        print(new_position, new_orientation)
    '''.strip().replace('\n    ', '\n'),

    "close_gripper": '''
    Close the gripper.
    Will not feedback whether the gripper was open before. Make sure it is.

    Returns:
        None

    Example:
        close_gripper()
    '''.strip().replace('\n    ', '\n'),

    "open_gripper": '''
    Open the gripper.
    Will not feedback whether the gripper was closed before. Make sure it is.
    
    Returns:
        None
    '''.strip().replace('\n    ', '\n'),

    "put_first_on_second": '''
    This is a helper function to move an object in one position to another.

    Args:
        pick_pos (list): The [x, y, z] position of the object to move, if z is not given it will default to a point near the table.
        place_pos (list): The [x, y, z] position to move the object to, similarly if z is not given it will default to a low point.
        pick_angle (float) = 0: When picking up the object, you can optionally specify the angle to rotate the gripper to in order to get the best grasp.

    Returns:
        Success: True if the object was successfully moved.

    Example:
        success = put_first_on_second(block_pos, [block_pos[0] + 0.1, block_pos[1], block_pos[2]])
        print("moving block to the right: ", success)
        for (teabag, mug) in zip(teabag_positions, mug_positions):
            success = put_first_on_second(teabag, mug)
            print("moving teabag to mug: ", success)
    '''.strip().replace('\n    ', '\n'),
}

final_system_message = '''
This is the end of the examples. The real user's query will follow. All previous variables no longer exist and you must start from scratch. The demonstration shows you only a very limited way to interact with the environment. You should use your creativity to adapt and solve the user's task.
'''.strip().replace('\n    ', '\n')

check_completion_message = '''
This is a standard check to ensure that the user's task is complete. It is very possible that it is truly complete.
The original task was:
"{query}"
An image of the final scene is provided below. First, reason about the user's task and the current state of the scene. Then, if the task is complete you should repeat the COMPLETE tool in your response. If it is not, provide feedback to the robot agent to let it know what is missing.
You MUST try something as new as possible rather than repeating the same actions.
If in doubt, you may also find the vision agent useful for confirmation whether a task is complete. For example, you can get the new positions of objects you think you have moved and see if they are now in the correct positions (try to consider the new positions relative to the old ones).
'''.strip().replace('\n    ', '\n')

missing_tool_use_correction = '''
Format error: No $$ found in completion. Please make sure you use exactly one of $$CODE$$, $$VIEW_SCENE$$, or $$COMPLETE$$ in your response.
'''

stack_blocks_all_modules = [
    '''
    The vision assistant's output meets my expectations, I will continue with step 1.
    I will now stack the blocks in order of size. Blocks are easy to pick up so I do not need to ask the vision assistant for how to grasp them.
    I will proceed carefully and check progress as I go.
    $$CODE$$
    largest_block = block_names[0] + ' block'
    middle_block = block_names[1] + ' block'
    smallest_block = block_names[2] + ' block'
    largest_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {largest_block}.")
    print("largest block position: ", largest_block_pos)
    middle_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {middle_block}.")
    print("middle block position: ", middle_block_pos)
    success = put_first_on_second(middle_block_pos, largest_block_pos)
    print(success)
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    largest block position: [0.102, -0.358, 0.116]
    middle block position: [0.062, -0.322, 0.043]
    True
    '''.strip().replace('\n    ', '\n'),

    '''
    I think the blocks were successfully stacked. I will now check the scene to verify.
    $$VIEW_SCENE$$
    '''.strip().replace('\n    ', '\n'),

    '''
    Image redacted in example. Image shows that the green block has moved slightly but is not stacked on the red block.
    '''.strip().replace('\n    ', '\n'),

    '''
    The blocks are not stacked as expected. I try to more carefully pick up the middle block.
    I will think of possible next things to try systematically:
    1. The vision assistant can help identify the best angle to grasp objects at, and I could use that angle in the put_first_on_second function.
    2. I could hover over the block before grasping it to ensure I am in the correct position.
    3. I can check more frequently with the VIEW_SCENE tool to ensure the movement is working as expected.
    4. I could try using fine control to pick up the block from the side. The vision assistant may be able to give me the dimensions of the block to help with this.

    I think my most promising option is to use the vision assistant to get the best angle to grasp the middle block and pause after picking up the block to check it has been grasped correctly.
    $$CODE$$
    middle_block_grasp_pos, middle_block_grasp_z_angle = vision_assistant(f"Please return the position and z angle of how to grasp the {middle_block}. Return a tuple of a list and a float: [x, y, z], z_angle. The {middle_block} was last located at approximately (0.062, -0.322, 0.043) but may have moved since.")
    print('middle block grasp: ', middle_block_grasp_pos, middle_block_grasp_z_angle)
    # I will hover over the block before grasping it.
    hover_pos = [middle_block_grasp_pos[0], middle_block_grasp_pos[1], middle_block_grasp_pos[2] + 0.1]
    # I will print out the results to check the robot is successfully moving to each point.
    print("hover: ", move_robot(hover_pos, [0, 0, middle_block_grasp_z_angle]))
    print("grasp pos: ", move_robot(middle_block_grasp_pos, [0, 0, middle_block_grasp_z_angle]))
    print("gripper: ", close_gripper())
    print("pick up: ", move_robot(hover_pos, [0, 0, middle_block_grasp_z_angle]))
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    middle block grasp: [0.068, -0.312, 0.043] [0.0, 0.0, 0.589]
    hover:  ([0.069, -0.312, 0.144], [-0.004, 0.002, 0.590])
    grasp pos:  ([0.070, -0.311, 0.047], [-0.004, 0.001, 0.589])
    Gripper now closed.
    gripper: None
    pick up:  ([0.069, -0.311, 0.142], [-0.004, 0.002, 0.590])
    '''.strip().replace('\n    ', '\n'),

    '''
    All the movements were successful, I now expect to have the middle block in the gripper.
    I will verify this by checking the scene.
    $$VIEW_SCENE$$
    '''.strip().replace('\n    ', '\n'),

    '''
    Image redacted in example. Image shows the middle block in the gripper.
    '''.strip().replace('\n    ', '\n'),

    '''
    The middle block is now in the gripper. I will now place it on the largest block. Since the block is already in my gripper I cannot use the put_first_on_second function. I will move the gripper to the position of the largest block and then move it down slightly to place the middle block.
    I am still in step 1, stacking the blocks steadily in order of size.
    I must place the middle block on the largest one.
    $$CODE$$
    largest_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {largest_block}. It was last seen at approximately (0.102, -0.358, 0.116) but may have moved since.")
    print("largest block position: ", largest_block_pos)
    # I will hover over the block before placing the middle block, I must account for the size of the block in my gripper too.
    hover_pos = [largest_block_pos[0], largest_block_pos[1], largest_block_pos[2] + 0.1]
    place_pos = [largest_block_pos[0], largest_block_pos[1], largest_block_pos[2] + 0.05]
    # I will print out the results to check the robot is successfully moving to each point.
    print("hover: ", move_robot(hover_pos, [0, 0, 0]))
    print("place pos: ", move_robot(place_pos, [0, 0, 0]))
    print("gripper: ", open_gripper())
    print("return to hover: ", move_robot(hover_pos, [0, 0, 0]))
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    largest block position: [0.102, -0.358, 0.116]
    hover:  ([0.102, -0.358, 0.216], [0.0, 0.0, 0.0])
    place pos:  ([0.102, -0.358, 0.166], [0.0, 0.0, 0.0])
    Gripper now open.
    gripper: None
    return to hover:  ([0.102, -0.358, 0.216], [0.0, 0.0, 0.0])
    '''.strip().replace('\n    ', '\n'),

    '''
    The middle block was successfully placed on the largest block. I will now check the scene to verify.
    $$VIEW_SCENE$$
    '''.strip().replace('\n    ', '\n'),

    '''
    Image redacted in example. Image shows the middle block successfully placed on the largest block.
    '''.strip().replace('\n    ', '\n'),

    '''
    The middle block is now placed on the largest block. I am still in step 1, stacking the blocks steadily in order of size.
    I must now place the smallest block on the middle block.
    $$CODE$$
    smallest_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {smallest_block}.")
    print("smallest block position: ", smallest_block_pos)
    middle_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {middle_block}.")
    print("middle block position: ", middle_block_pos)
    success = put_first_on_second(smallest_block_pos, middle_block_pos)
    print(success)
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    smallest block position: [-0.109, -0.292, 0.033]
    middle block position: [0.097, -0.359, 0.132]
    True
    '''.strip().replace('\n    ', '\n'),
]

stack_blocks_no_fine_control = [
    '''
    The vision assistant's output meets my expectations, I will continue with step 1.
    I will now stack the blocks in order of size. Blocks are easy to pick up so I do not need to ask the vision assistant for how to grasp them.
    I will proceed carefully and check progress as I go.
    $$CODE$$
    largest_block = block_names[0] + ' block'
    middle_block = block_names[1] + ' block'
    smallest_block = block_names[2] + ' block'
    largest_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {largest_block}.")
    print("largest block position: ", largest_block_pos)
    middle_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {middle_block}.")
    print("middle block position: ", middle_block_pos)
    success = put_first_on_second(middle_block_pos, largest_block_pos)
    print(success)
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    largest block position: [0.102, -0.358, 0.116]
    middle block position: [0.062, -0.322, 0.043]
    True
    '''.strip().replace('\n    ', '\n'),

    '''
    I think the blocks were successfully stacked. I will now check the scene to verify.
    $$VIEW_SCENE$$
    '''.strip().replace('\n    ', '\n'),

    '''
    Image redacted in example. Image shows that the green block has moved slightly but is not stacked on the red block.
    '''.strip().replace('\n    ', '\n'),

    '''
    The blocks are not stacked as expected. I try to more carefully pick up the middle block.
    The vision assistant can help identify the best angle to grasp objects at.
    $$CODE$$
    middle_block_grasp_pos, middle_block_grasp_z_angle = vision_assistant(f"Please return the position and z angle of how to grasp the {middle_block}. Return a tuple of a list and a float: [x, y, z], z_angle. The {middle_block} was last located at approximately (0.062, -0.322, 0.043) but may have moved since.")
    print('middle block grasp: ', middle_block_grasp_pos, middle_block_grasp_z_angle)
    success = put_first_on_second(middle_block_grasp_pos, largest_block_pos, middle_block_grasp_z_angle)
    print(success)
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    middle block grasp: [0.068, -0.312, 0.043] [0.0, 0.0, 0.589]
    True
    '''.strip().replace('\n    ', '\n'),

    '''
    I think the pick and place was successful. I will now check the scene to verify.
    $$VIEW_SCENE$$
    '''.strip().replace('\n    ', '\n'),

    '''
    Image redacted in example. Image shows green block on top of a red block.
    '''.strip().replace('\n    ', '\n'),

    '''
    The middle block is now placed on the largest block. I am still in step 1, stacking the blocks steadily in order of size.
    I must now place the smallest block on the middle block.
    $$CODE$$
    smallest_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {smallest_block}.")
    print("smallest block position: ", smallest_block_pos)
    middle_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {middle_block}.")
    print("middle block position: ", middle_block_pos)
    success = put_first_on_second(smallest_block_pos, middle_block_pos)
    print(success)
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    smallest block position: [-0.109, -0.292, 0.033]
    middle block position: [0.097, -0.359, 0.132]
    True
    '''.strip().replace('\n    ', '\n'),
]

stack_blocks_no_pick_and_place = [
    '''
    The vision assistant's output meets my expectations, I will continue with step 1.
    I will now stack the blocks in order of size. Blocks are easy to pick up so I do not need to ask the vision assistant for how to grasp them.
    I will proceed carefully and check progress as I go.
    $$CODE$$
    largest_block = block_names[0] + ' block'
    middle_block = block_names[1] + ' block'
    smallest_block = block_names[2] + ' block'
    largest_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {largest_block}.")
    print("largest block position: ", largest_block_pos)
    middle_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {middle_block}.")
    print("middle block position: ", middle_block_pos)
    
    # I will hover over the block before grasping it.
    hover_pos = [middle_block_pos[0], middle_block_pos[1], middle_block_pos[2] + 0.1]
    print("hover: ", move_robot(hover_pos, [0, 0, 0]))
    print("pick pos: ", move_robot(middle_block_pos, [0, 0, 0]))
    print("gripper: ", close_gripper())
    print("pick up: ", move_robot(hover_pos, [0, 0, 0]))
    # i will hover over the largest block before placing the smallest block on it
    hover_pos = [largest_block_pos[0], largest_block_pos[1], largest_block_pos[2] + 0.1]
    print("hover: ", move_robot(hover_pos, [0, 0, 0]))
    print("place pos: ", move_robot(largest_block_pos, [0, 0, 0]))
    print("gripper: ", open_gripper())
    print("move away: ", move_robot(hover_pos, [0, 0, 0]))
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    largest block position: [0.102, -0.358, 0.116]
    middle block position: [0.062, -0.322, 0.043]
    hover:  ([0.062, -0.322, 0.143], [0.0, 0.0, 0.0])
    place pos:  ([0.062, -0.322, 0.043], [0.0, 0.0, 0.0])
    Gripper now open.
    gripper: None
    return to hover:  ([0.062, -0.322, 0.143], [0.0, 0.0, 0.0])
    hover:  ([0.102, -0.358, 0.216], [0.0, 0.0, 0.0])
    place pos:  ([0.102, -0.358, 0.166], [0.0, 0.0, 0.0])
    Gripper now open.
    gripper: None
    return to hover:  ([0.102, -0.358, 0.216], [0.0, 0.0, 0.0])
    '''.strip().replace('\n    ', '\n'),

    '''
    I think the blocks were successfully stacked. I will now check the scene to verify.
    $$VIEW_SCENE$$
    '''.strip().replace('\n    ', '\n'),

    '''
    Image redacted in example. Image shows that the green block has moved slightly but is not stacked on the red block.
    '''.strip().replace('\n    ', '\n'),

    '''
    The blocks are not stacked as expected. I try to more carefully pick up the middle block.
    The vision assistant can help identify the best angle to grasp objects at.
    $$CODE$$
    middle_block_grasp_pos, middle_block_grasp_z_angle = vision_assistant(f"Please return the position and z angle of how to grasp the {middle_block}. Return a tuple of a list and a float: [x, y, z], z_angle. The {middle_block} was last located at approximately (0.062, -0.322, 0.043) but may have moved since.")
    print('middle block grasp: ', middle_block_grasp_pos, middle_block_grasp_z_angle)
    # I will hover over the block before grasping it.
    hover_pos = [middle_block_grasp_pos[0], middle_block_grasp_pos[1], middle_block_grasp_pos[2] + 0.1]
    # I will print out the results to check the robot is successfully moving to each point.
    print("hover: ", move_robot(hover_pos, [0, 0, middle_block_grasp_z_angle]))
    print("grasp pos: ", move_robot(middle_block_grasp_pos, [0, 0, middle_block_grasp_z_angle]))
    print("gripper: ", close_gripper())
    print("pick up: ", move_robot(hover_pos, [0, 0, middle_block_grasp_z_angle]))
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    middle block grasp: [0.068, -0.312, 0.043] [0.0, 0.0, 0.589]
    hover:  ([0.069, -0.312, 0.144], [-0.004, 0.002, 0.590])
    grasp pos:  ([0.070, -0.311, 0.047], [-0.004, 0.001, 0.589])
    Gripper now closed.
    gripper: None
    pick up:  ([0.069, -0.311, 0.142], [-0.004, 0.002, 0.590])
    '''.strip().replace('\n    ', '\n'),

    '''
    All the movements were successful, I now expect to have the middle block in the gripper.
    I will verify this by checking the scene.
    $$VIEW_SCENE$$
    '''.strip().replace('\n    ', '\n'),

    '''
    Image redacted in example. Image shows the middle block in the gripper.
    '''.strip().replace('\n    ', '\n'),

    '''
    The middle block is now in the gripper. I will now place it on the largest block. Since the block is already in my gripper I cannot use the put_first_on_second function. I will move the gripper to the position of the largest block and then move it down slightly to place the middle block.
    I am still in step 1, stacking the blocks steadily in order of size.
    I must place the middle block on the largest one.
    $$CODE$$
    largest_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {largest_block}. It was last seen at approximately (0.102, -0.358, 0.116) but may have moved since.")
    print("largest block position: ", largest_block_pos)
    # I will hover over the block before placing the middle block, I must account for the size of the block in my gripper too.
    hover_pos = [largest_block_pos[0], largest_block_pos[1], largest_block_pos[2] + 0.1]
    place_pos = [largest_block_pos[0], largest_block_pos[1], largest_block_pos[2] + 0.05]
    # I will print out the results to check the robot is successfully moving to each point.
    print("hover: ", move_robot(hover_pos, [0, 0, 0]))
    print("place pos: ", move_robot(place_pos, [0, 0, 0]))
    print("gripper: ", open_gripper())
    print("return to hover: ", move_robot(hover_pos, [0, 0, 0]))
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    largest block position: [0.102, -0.358, 0.116]
    hover:  ([0.102, -0.358, 0.216], [0.0, 0.0, 0.0])
    place pos:  ([0.102, -0.358, 0.166], [0.0, 0.0, 0.0])
    Gripper now open.
    gripper: None
    return to hover:  ([0.102, -0.358, 0.216], [0.0, 0.0, 0.0])
    '''.strip().replace('\n    ', '\n'),

    '''
    The middle block was successfully placed on the largest block. I will now check the scene to verify.
    $$VIEW_SCENE$$
    '''.strip().replace('\n    ', '\n'),

    '''
    Image redacted in example. Image shows the middle block successfully placed on the largest block.
    '''.strip().replace('\n    ', '\n'),

    '''
    The middle block is now placed on the largest block. I am still in step 1, stacking the blocks steadily in order of size.
    I must now place the smallest block on the middle block.
    $$CODE$$
    smallest_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {smallest_block}.")
    print("smallest block position: ", smallest_block_pos)
    middle_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {middle_block}.")
    print("middle block position: ", middle_block_pos)

    # I will hover over the block before grasping it.
    hover_pos = [smallest_block_pos[0], smallest_block_pos[1], smallest_block_pos[2] + 0.1]
    print("hover: ", move_robot(hover_pos, [0, 0, 0]))
    print("pick pos: ", move_robot(smallest_block_pos, [0, 0, 0]))
    print("gripper: ", close_gripper())
    print("pick up: ", move_robot(hover_pos, [0, 0, 0]))
    # i will hover over the largest block before placing the smallest block on it
    hover_pos = [middle_block_pos[0], middle_block_pos[1], middle_block_pos[2] + 0.1]
    print("hover: ", move_robot(hover_pos, [0, 0, 0]))
    print("place pos: ", move_robot(middle_block_pos, [0, 0, 0]))
    print("gripper: ", open_gripper())
    print("move away: ", move_robot(hover_pos, [0, 0, 0]))
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    smallest block position: [-0.109, -0.292, 0.033]
    middle block position: [0.097, -0.359, 0.132]
    hover:  ([-0.109, -0.292, 0.133], [0.0, 0.0, 0.0])
    place pos:  ([-0.109, -0.292, 0.033], [0.0, 0.0, 0.0])
    Gripper now open.
    gripper: None
    return to hover:  ([-0.109, -0.292, 0.133], [0.0, 0.0, 0.0])
    hover:  ([0.097, -0.359, 0.232], [0.0, 0.0, 0.0])
    place pos:  ([0.097, -0.359, 0.132], [0.0, 0.0, 0.0])
    Gripper now open.
    gripper: None
    return to hover:  ([0.097, -0.359, 0.232], [0.0, 0.0, 0.0])
    '''.strip().replace('\n    ', '\n'),
]

knock_over_all_modules = [
    '''
    There is an egg in the scene. I am now finished with step 2.
    I will now move on to step 3, knocking the blocks over.
    Expectations for completion: the blocks should be knocked over and I should be able to see that they are.
    The blocks may have moved slightly when I placed the smallest block on the middle block, I will find the position of the middle block and knock the blocks over from there.
    I should use fine control for this as it is not a simple pick and place task. A closed gripper will be best for pushing.
    $$CODE$$
    middle_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {middle_block}. It might be underneath the {smallest_block}. Return only the position of the {middle_block}.")
    print("middle block position: ", middle_block_pos)
    # I will move the gripper to the side of the middle block and then move it across to knock the blocks over.
    side_pos = [middle_block_pos[0] + 0.1, middle_block_pos[1], middle_block_pos[2]]
    # I will print out the results to check the robot is successfully moving to each point.
    print("gripper: ", close_gripper())
    print("side pos: ", move_robot(side_pos, [0, 0, 0]))
    print("knock over: ", move_robot_relative([0.2, 0, 0], [0, 0, 0]))
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    middle block position:  [0.097, -0.359, 0.132]
    Gripper now closed.
    gripper: None
    side pos:  ([0.197, -0.359, 0.132], [0.0, 0.0, 0.0])
    knock over:  ([0.397, -0.359, 0.132], [0.0, 0.0, 0.0])
    '''.strip().replace('\n    ', '\n'),

    '''
    Looking at the positions the robot moved to, I made a mistake in the relative movement.
    I am still on step 3, knocking the blocks over.
    I will try again.
    $$CODE$$
    print("side pos: ", move_robot(side_pos, [0, 0, 0]))
    print("knock over: ", move_robot_relative([-0.2, 0, 0], [0, 0, 0]))
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    side pos:  ([0.197, -0.359, 0.132], [0.0, 0.0, 0.0])
    knock over:  ([-0.003, -0.359, 0.132], [0.0, 0.0, 0.0])
    '''.strip().replace('\n    ', '\n'),
]

knock_over_no_fine_control = [
    '''
    There is an egg in the scene. I am now finished with step 2.
    I will now move on to step 3, knocking the blocks over.
    Expectations for completion: the blocks should be knocked over and I should be able to see that they are.
    The blocks may have moved slightly when I placed the smallest block on the middle block, I will find the position of the middle block and knock the blocks over from there.
    I will have to be imaginative in how I knock the blocks over with my control primitive put_first_on_second.
    Possible ideas that would work would attempting to pick up the middle or bottom block, or perhaps deconstructing the pile from the top.
    $$CODE$$
    middle_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {middle_block}. It might be underneath the {smallest_block}. Return only the position of the {middle_block}.")
    print("middle block position: ", middle_block_pos)
    # I will try picking up the middle block and 'placing' it above the tower.
    above_middle_block_pos = [middle_block_pos[0], middle_block_pos[1], middle_block_pos[2] + 0.1]
    success = put_first_on_second(middle_block_pos, above_middle_block_pos)
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    middle block position:  [0.097, -0.359, 0.132]
    True
    '''.strip().replace('\n    ', '\n'),
]

# user, [assistant, system]*
def construct_example(stack_blocks, knock_over):
    return [
    '''
    Please stack the blocks steadily and then, when you see an egg, knock them over.
    '''.strip().replace('\n    ', '\n'),

    gptv_injection_message,

    '''
    Image redacted in example. Image shows scene with 3 blocks of different sizes.
    '''.strip().replace('\n    ', '\n'),

    '''
    Contextual observation: There are three blocks and they are each different sizes. However, all three are small enough they should be easy to pick up with a vertical gripper.
    There is a one red, one blue and one green block. I will not try to judge the relative sizes from the image and will ask the vision_assistant to help.

    Broad objectives:
    1. Stack the blocks steadily. Stack them in order of size.
    2. Wait to see an egg.
    3. Knock the blocks over when you see the egg.

    I will now begin step 1, stacking the blocks steadily in order of size.
    Expectations for completion: all three blocks should have similar x, y positions at the end and I should be able to see that they are stacked by viewing the scene.

    Within step 1:
    I will first order the three blocks by size.
    $$CODE$$
    block_names = vision_assistant("Please order the three blocks by size, largest first. Please return a list of strings with the names of the blocks. I believe there is one each of a red, blue, and green block.")
    print("block names: "block_names)
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    block names: ['red', 'green', 'blue']
    '''.strip().replace('\n    ', '\n'),

    *stack_blocks,

    '''
    I think the blocks were successfully stacked. I will now check the scene to verify.
    $$VIEW_SCENE$$
    '''.strip().replace('\n    ', '\n'),

    '''
    Image redacted in example. Image shows the smallest block successfully placed on the middle block.
    '''.strip().replace('\n    ', '\n'),

    '''
    The blocks are now stacked in order of size. I am now finished with step 1.
    I will now move on to step 2, waiting to see an egg.
    Expectations for completion: the vision agent should detect and egg in the scene and I should be able to confirm it.
    I should set a reasonable time after which to check that my method is correct.
    $$CODE$$
    import time
    egg = False
    count = 0
    while not egg and count < 10:
        egg = vision_assistant("Please return if an egg is in the scene. Return a single boolean. There may be no egg at all.")
        print(egg)
        time.sleep(5)
        count += 1
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    False
    False
    False
    False
    False
    True
    '''.strip().replace('\n    ', '\n'),

    '''
    I think there is an egg in the scene. I will now check the scene to verify.
    $$VIEW_SCENE$$
    '''.strip().replace('\n    ', '\n'),

    '''
    Image redacted in example. Image shows an egg in the scene.
    '''.strip().replace('\n    ', '\n'),

    *knock_over,

    '''
    I think the blocks were successfully knocked over. I will now check the scene to verify.
    $$VIEW_SCENE$$
    '''.strip().replace('\n    ', '\n'),

    '''
    Image redacted in example. Image shows the blocks knocked over.
    Remember to actually judge images in the real run as the robot is very unreliable or you might have controlled it incorrectly.
    '''.strip().replace('\n    ', '\n'),

    '''
    The blocks are now knocked over. I am now finished with step 3.
    The user's original task was to stack the blocks steadily and then, when you see an egg, knock them over.
    I have completed the task and verified the scene. I will now give a final message.
    $$COMPLETE$$
    '''.strip().replace('\n    ', '\n'),
]

all_modules_example = construct_example(stack_blocks_all_modules, knock_over_all_modules)
no_pick_and_place_example = construct_example(stack_blocks_no_pick_and_place, knock_over_all_modules)
no_fine_control_example = construct_example(stack_blocks_no_fine_control, knock_over_no_fine_control)


grasping_example_snippet = [
    '''
    This is a partial example, to show important functionality of the vision assistant and movement tools.
    The user's original task was to pass a cok. However when inspecting the scene you could not see a coke can, only a set of drawers and a piece of paper lying flat with something underneath.
    When completing step 1 of your broad plan, to look for the can, your first attempt was to open the drawer using the vision assistant to know where to grasp and the move_robot function to open the drawer.
    This was unsuccessful as there was no can inside so you continued with step 1 and decided to move the paper. The example begins here.
    '''.strip().replace('\n    ', '\n'),

    '''
    There is no coke can in the drawer. I will continue with step 1 as my expectations for completion have not been met.
    I will now move the paper to see if the coke can is underneath.
    A flat piece of paper, like many objects including things with handles, long objects, and objects with a flat surface, cannot be grasped simply by grasping across the middle and therefore is hard to grasp.
    The vision assistant can help identify the best position and angle to grasp objects at.
    If this does not work, I may have to creatively control the robot to move the paper.
    $$CODE$$
    paper_grasp_pos, paper_grasp_z_angle = vision_assistant("Please return the position and z angle of how to grasp the paper. Return a tuple of a list and a float: [x, y, z], z_angle.")
    print('paper grasp: ', paper_grasp_pos, paper_grasp_z_angle)

    # I will hover over the paper before grasping it.
    hover_pos = [paper_grasp_pos[0], paper_grasp_pos[1], paper_grasp_pos[2] + 0.1]
    # I will print out the results to check the robot is successfully moving to each point.
    print("hover: ", move_robot(hover_pos, [0, 0, paper_grasp_z_angle]))
    print("grasp pos: ", move_robot(paper_grasp_pos, [0, 0, paper_grasp_z_angle]))
    print("gripper: ", close_gripper())
    print("pick up: ", move_robot(hover_pos, [0, 0, paper_grasp_z_angle]))
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    paper grasp: [0.068, -0.312, 0.043] [0.0, 0.0, 0.589]
    hover:  ([0.069, -0.312, 0.144], [-0.004, 0.002, 0.590])
    grasp pos:  ([0.070, -0.311, 0.047], [-0.004, 0.001, 0.589])
    Gripper now closed.
    gripper: None
    pick up:  ([0.069, -0.311, 0.142], [-0.004, 0.002, 0.590])
    '''.strip().replace('\n    ', '\n'),

    '''
    All the movements were successful, I now expect to have the paper in the gripper.
    I will verify this by checking the scene.
    $$VIEW_SCENE$$
    '''.strip().replace('\n    ', '\n'),

    '''
    Image redacted in example. Image shows the still covering something.
    '''.strip().replace('\n    ', '\n'),

    '''
    The paper has not been moved, I shall try creatively gripping the paper.
    The paper is flat, I should angle the gripper to try to grasp it from the side.
    I am not sure which side I am currently attempting to grasp, let me pause and check the scene.
    $$CODE$$
    print("gripper: ", open_gripper())
    print("hover: ", move_robot(hover_pos, [0, 0, paper_grasp_z_angle]))
    print("grasp pos: ", move_robot(paper_grasp_pos, [0, 0, paper_grasp_z_angle]))
    print("rotate positive x: ", move_robot_relative([0, 0, 0], [0.2, 0, 0]))
    print("gripper: ", close_gripper())
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    Gripper now open.
    gripper: None
    hover:  ([0.069, -0.311, 0.142], [-0.004, 0.002, 0.590])
    grasp pos:  ([0.070, -0.310, 0.047], [-0.004, 0.001, 0.589])
    rotate positive y:  ([0.070, -0.310, 0.047], [0.196, 0.001, 0.589])
    Gripper now closed.
    '''.strip().replace('\n    ', '\n'),

    '''
    I need to check whether the paper was grasped.
    $$VIEW_SCENE$$
    '''.strip().replace('\n    ', '\n'),

    '''
    Image redacted in example. Image shows the gripper at the far edge of the paper but tilted slightly to the side.
    '''.strip().replace('\n    ', '\n'),

    '''
    The paper was not successfully grasped, and this angle of rotation did not seem to help.
    Also, I can also notice that the gripper is quite high for grasping the paper edge.
    I will try a different angle of rotation and move slightly downwards.
    $$CODE$$
    print("open gripper: ", open_gripper())
    print("grasp_pos: ", move_robot(grasp_pos, [0, 0, paper_grasp_z_angle]))
    print("rotate positive y and slightly down: ", move_robot_relative([0, 0, -0.01], [0, 0.2, 0]))
    print("gripper: ", close_gripper())
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    open gripper: None
    grasp_pos:  ([0.070, -0.310, 0.047], [-0.004, 0.001, 0.589])
    rotate positive y and slightly down:  ([0.070, -0.310, 0.037], [-0.004, 0.201, 0.589])
    Gripper now closed.
    '''.strip().replace('\n    ', '\n'),

    '''
    I need to check whether the paper was grasped.
    $$VIEW_SCENE$$
    '''.strip().replace('\n    ', '\n'),

    '''
    Image redacted in example. Image shows the gripper at the edge of the paper and slightly tilted towards the camera.
    '''.strip().replace('\n    ', '\n'),

    '''
    The paper was still not successfully grasped as the gripper tilted away from the edge instead of towards it.
    From this i can learn that y is the correct axis to rotate but it should be negative instead of positive.
    I will move downwards in a similar way but rotate in the other direction.
    $$CODE$$
    print("open gripper: ", open_gripper())
    print("grasp_pos: ", move_robot(grasp_pos, [0, 0, paper_grasp_z_angle]))
    print("rotate negative y and slightly down: ", move_robot_relative([0, 0, -0.01], [0, -0.2, 0]))
    print("gripper: ", close_gripper())
    '''.strip().replace('\n    ', '\n'),

    '''
    stdout:
    open gripper: None
    grasp_pos:  ([0.070, -0.310, 0.047], [-0.004, 0.001, 0.589])
    rotate negative y and slightly down:  ([0.070, -0.310, 0.037], [-0.004, -0.199, 0.589])
    Gripper now closed.
    '''.strip().replace('\n    ', '\n'),

    '''
    I need to check whether the paper was grasped.
    $$VIEW_SCENE$$
    '''.strip().replace('\n    ', '\n'),

    '''
    Image redacted in example. Image shows the gripper grasping the edge of the paper and slightly tilted away from the camera.
    '''.strip().replace('\n    ', '\n'),

    '''
    The paper was successfully grasped. I will now move it away to continue with step 1, looking for the coke can.
    I am expecting to see a coke can by the end of step 1, and then I can move onto step 2 to complete the user's original task of passing the coke.
    $$CODE$$
    print("list paper up: ", move_robot_relative([0, 0, 0.2], [0, 0, 0]))
    '''.strip().replace('\n    ', '\n'),

    '''
    This is the end of this partial example, to show important functionality of the vision assistant and movement tools.
    Remember that this is not all you can do, be creative and relentless in solving problems.
    '''.strip().replace('\n    ', '\n'),
]
