top_system_message = '''
You are a careful, experienced agent operating a robot in a real-world environment.
Any code snippets after you mark the $$CODE$$ tool will be sent to be executed on the robot, and you will receive back any printed stdout.
Alternatively, you can end your message with the $$VIEW_SCENE$$ tool to request to see an image of the scene. You must use use this only to verify the existence of objects, gain contextual information, or to verify progress.
Once you have checked that the user's entire task is complete, you should give a final message ending with $$COMPLETE$$.

When solving a task, you should first outline the broad stages of the task. Then, when starting each stage of the task, remind yourself of the goal of the stage. Set youself expectations for the completion of this stage and make sure you achieve them before moving on.
Verify progress by printing the output of various functions and displaying images.
Be prepared to adapt to the environment as you may discover important context as you progress through the task.

Setup information:
All positions are in meters and all angles are in radians. Positive x is to the right, positive y is forward, and positive z is up. The robot's gripper is vertical when the xrot and yrot are 0, zrot controls the angle of the wrist. xrot tilts the gripper forward, yrot tilts the gripper sideways.

Within the system, functions are defined for you to interact with the environment.
{functions_advice}
The functions have the following signatures:
{functions_docs}

The robot system has the following packages available:
{packages}

{few_shot_introduction}
'''.strip()

few_shot_introduction = '''
What follows are some example interactions between you and the user. Please follow the same format.
'''.strip()

gptv_injection_message = '''
Before outlining my broad objectives in order to complete the user's task, I should carefully check the current environment for any contextual information. I will use the VIEW_SCENE tool to request an image of the scene.
$$VIEW_SCENE$$
'''.strip()

functions_advice = {
    "vision_assistant": "the vision_assistant() function allows you to delegate perception tasks to an agent who has access to perception models. You can request information about the scene and specify the requested information and return format in the message. You can also request information in the format of a string and print it in order to inform your own decisions.",
    # If move robot is in the other primitive move functions are also in
    "move_robot": "You can finely control the robot's gripper using the move_robot and gripper functions. When you do not specify an orientation, the gripper will default to a vertical orientation in the absolute movement and it will not change the orientation in the relative movement. In order to only change the orientation, you can use the move_robot_relative function with a position of [0, 0, 0]. While using these controls, frequently remind yourself of your overall aim and check progress against it using VIEW_SCENE.",
    "put_first_on_second": "This function is a helper, it usefully abstracts fine control for when an object is easy to pick up and move. You should use it when you can but if it doesn't work consider controlling the robot more finely with the move_robot function."
}

function_docs = {
    "vision_assistant": '''
    Make a request to an agent who has access to perception models to get information about the scene. Specify the requested information and return format in the message.

    Args:
        message (str): The message to send to the vision assistant, be very clear about the information you need and the format you need it in. You should also provide other contextual information that will help, such as the current objects you know about in the environment.

    Returns:
        A response of the type and format requested.

    Example:
        cup_xyz = vision_assistant("Please return the [x, y, z] position of the cup in the scene.")
        print(cup_xyz)
        left_most_bowl = vision_assistant("Please tell me which bowl is largest in the scene. There are three bowls: two white and one red. Please return a string, such as 'leftmost white bowl' or 'white bowl at position (x, y, z)'.")
        print(left_most_bowl)
        mug_grasp_position, mug_grasp_z_angle = vision_assistant("Please return the position and z angle of how to grasp the ceramic mug. Return a tuple of a list and a float: [x, y, z], z_angle.")
    ''',

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
    ''',
    
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
    ''',

    "close_gripper": '''
    Close the gripper.

    Returns:
        None

    Example:
        close_gripper()
    ''',

    "open_gripper": '''
    Open the gripper.
    
    Returns:
        None
    ''',

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
    '''
}

final_system_message = '''
This is the end of the examples. The real user's query will follow. All previous variables no longer exist and you must start from scratch.
'''.strip()

check_completion_message = '''
This is a standard check to ensure that the user's task is complete. It may truly be complete.
The original task was:
"{query}"
An image of the final scene is provided below. First, reason about the user's task and the current state of the scene. Then, if the task is complete you should repeat the COMPLETE tool in your response. If it is not, provide feedback to the robot agent to let it know what is missing.
'''.strip()

missing_tool_use_correction = '''
Format error: No $$ found in completion. Please make sure you use exactly one of $$CODE$$, $$VIEW_SCENE$$, or $$COMPLETE$$ in your response.
'''

# user, [assistant, system]*
all_modules_example = [
    '''
    Please stack the blocks steadily and then, when you see an egg, knock them over.
    '''.strip(),

    gptv_injection_message,

    '''
    Image redacted in example. Image shows scene with 3 blocks of different sizes.
    '''.strip(),

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
    '''.strip(),

    '''
    stdout:
    block names: ['red', 'green', 'blue']
    '''.strip(),

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
    '''.strip(),

    '''
    stdout:
    largest block position: [0.102, -0.358, 0.116]
    middle block position: [0.062, -0.322, 0.043]
    True
    '''.strip(),

    '''
    I think the blocks were successfully stacked. I will now check the scene to verify.
    $$VIEW_SCENE$$
    '''.strip(),

    '''
    Image redacted in example. Image shows that the green block has moved slightly but is not stacked on the red block.
    '''.strip(),

    '''
    The blocks are not stacked as expected. I try to more carefully pick up the middle block.
    $$CODE$$
    middle_block_grasp_pos, middle_block_grasp_z_angle = vision_assistant(f"Please return the position and z angle of how to grasp the {middle_block}. Return a tuple of a list and a float: [x, y, z], z_angle.")
    print('middle block grasp: ', middle_block_grasp_pos, middle_block_grasp_z_angle)
    # I will hover over the block before grasping it.
    hover_pos = [middle_block_grasp_pos[0], middle_block_grasp_pos[1], middle_block_grasp_pos[2] + 0.1]
    # I will print out the results to check the robot is successfully moving to each point.
    print("hover: ", move_robot(hover_pos, [0, 0, middle_block_grasp_z_angle]))
    print("grasp pos: ", move_robot(middle_block_grasp_pos, [0, 0, middle_block_grasp_z_angle]))
    print("gripper: ", close_gripper())
    print("pick up: ", move_robot(hover_pos, [0, 0, middle_block_grasp_z_angle]))
    '''.strip(),

    '''
    stdout:
    middle block grasp: [0.068, -0.312, 0.043] [0.0, 0.0, 0.589]
    hover:  ([0.069, -0.312, 0.144], [-0.004, 0.002, 0.590])
    grasp pos:  ([0.070, -0.311, 0.047], [-0.004, 0.001, 0.589])
    Gripper now closed.
    gripper: None
    pick up:  ([0.069, -0.311, 0.142], [-0.004, 0.002, 0.590])
    '''.strip(),

    '''
    All the movements were successful, I now expect to have the middle block in the gripper.
    I will verify this by checking the scene.
    $$VIEW_SCENE$$
    '''.strip(),

    '''
    Image redacted in example. Image shows the middle block in the gripper.
    '''.strip(),

    '''
    The middle block is now in the gripper. I will now place it on the largest block. Since the block is already in my gripper I cannot use the put_first_on_second function. I will move the gripper to the position of the largest block and then move it down slightly to place the middle block.
    I am still in step 1, stacking the blocks steadily in order of size.
    I must place the middle block on the largest one.
    $$CODE$$
    largest_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {largest_block}.")
    print("largest block position: ", largest_block_pos)
    # I will hover over the block before placing the middle block, I must account for the size of the block in my gripper too.
    hover_pos = [largest_block_pos[0], largest_block_pos[1], largest_block_pos[2] + 0.1]
    place_pos = [largest_block_pos[0], largest_block_pos[1], largest_block_pos[2] + 0.05]
    # I will print out the results to check the robot is successfully moving to each point.
    print("hover: ", move_robot(hover_pos, [0, 0, 0]))
    print("place pos: ", move_robot(place_pos, [0, 0, 0]))
    print("gripper: ", open_gripper())
    print("return to hover: ", move_robot(hover_pos, [0, 0, 0]))
    '''.strip(),

    '''
    stdout:
    largest block position: [0.102, -0.358, 0.116]
    hover:  ([0.102, -0.358, 0.216], [0.0, 0.0, 0.0])
    place pos:  ([0.102, -0.358, 0.166], [0.0, 0.0, 0.0])
    Gripper now open.
    gripper: None
    return to hover:  ([0.102, -0.358, 0.216], [0.0, 0.0, 0.0])
    '''.strip(),

    '''
    The middle block was successfully placed on the largest block. I will now check the scene to verify.
    $$VIEW_SCENE$$
    '''.strip(),

    '''
    Image redacted in example. Image shows the middle block successfully placed on the largest block.
    '''.strip(),

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
    '''.strip(),

    '''
    stdout:
    smallest block position: [-0.109, -0.292, 0.033]
    middle block position: [0.097, -0.359, 0.132]
    True
    '''.strip(),

    '''
    I think the blocks were successfully stacked. I will now check the scene to verify.
    $$VIEW_SCENE$$
    '''.strip(),

    '''
    Image redacted in example. Image shows the smallest block successfully placed on the middle block.
    '''.strip(),

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
        egg = vision_assistant("Please return if an egg is in the scene. Return a single boolean.")
        print(egg)
        time.sleep(5)
        count += 1
    '''.strip(),

    '''
    stdout:
    False
    False
    False
    False
    False
    True
    '''.strip(),

    '''
    I think there is an egg in the scene. I will now check the scene to verify.
    $$VIEW_SCENE$$
    '''.strip(),

    '''
    Image redacted in example. Image shows an egg in the scene.
    '''.strip(),

    '''
    There is an egg in the scene. I am now finished with step 2.
    I will now move on to step 3, knocking the blocks over.
    Expectations for completion: the blocks should be knocked over and I should be able to see that they are.
    The blocks may have moved slightly when I placed the smallest block on the middle block, I will find the position of the middle block and knock the blocks over from there.
    I should use fine control for this as it is not a simple pick and place task. A closed gripper will be best for pushing.
    $$CODE$$
    middle_block_pos = vision_assistant(f"Please return the [x, y, z] position of the {middle_block}.")
    print("middle block position: ", middle_block_pos)
    # I will move the gripper to the side of the middle block and then move it across to knock the blocks over.
    side_pos = [middle_block_pos[0] + 0.1, middle_block_pos[1], middle_block_pos[2]]
    # I will print out the results to check the robot is successfully moving to each point.
    print("gripper: ", close_gripper())
    print("side pos: ", move_robot(side_pos, [0, 0, 0]))
    print("knock over: ", move_robot_relative([0.2, 0, 0], [0, 0, 0]))
    '''.strip(),

    '''
    stdout:
    middle block position:  [0.097, -0.359, 0.132]
    Gripper now closed.
    gripper: None
    side pos:  ([0.197, -0.359, 0.132], [0.0, 0.0, 0.0])
    knock over:  ([0.397, -0.359, 0.132], [0.0, 0.0, 0.0])
    '''.strip(),

    '''
    Looking at the positions the robot moved to, I made a mistake in the relative movement.
    I am still on step 3, knocking the blocks over.
    I will try again.
    $$CODE$$
    print("side pos: ", move_robot(side_pos, [0, 0, 0]))
    print("knock over: ", move_robot_relative([-0.2, 0, 0], [0, 0, 0]))
    '''.strip(),

    '''
    stdout:
    side pos:  ([0.197, -0.359, 0.132], [0.0, 0.0, 0.0])
    knock over:  ([-0.003, -0.359, 0.132], [0.0, 0.0, 0.0])
    '''.strip(),

    '''
    I think the blocks were successfully knocked over. I will now check the scene to verify.
    $$VIEW_SCENE$$
    '''.strip(),

    '''
    Image redacted in example. Image shows the blocks knocked over.
    '''.strip(),

    '''
    The blocks are now knocked over. I am now finished with step 3.
    The user's original task was to stack the blocks steadily and then, when you see an egg, knock them over.
    I have completed the task and verified the scene. I will now give a final message.
    $$COMPLETE$$
    '''.strip(),
]
