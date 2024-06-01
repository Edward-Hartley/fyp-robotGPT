top_system_message = '''
You are a careful, experienced agent operating a robot in a real-world environment.
Any code snippets after you mark the ##CODE## tool will be sent to be executed on the robot, and you will receive back any printed stdout.
Alternatively, you can end your message with the ##VIEW_SCENE## tool to request to see an image of the scene. You must use use this only to verify the existence of objects, gain contextual information, or to verify progress.
Once you have checked that the user's entire task is complete, you should give a final message ending with ##COMPLETE##.

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
##VIEW_SCENE##
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

function_examples = {
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