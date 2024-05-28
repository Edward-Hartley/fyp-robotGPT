tools = [
    {
    "name": "move_robot",
    "description": "Move the robot to a specified position and orientation. Orientation is optional, no orientation with relative=false will move it so the gripper is pointing down, and with relative=true it will not change the current orientation.",
    "parameters": {
        "type": "object",
        "properties": {
        "position": {
            "type": "array",
            "items": {
            "type": "number"
            },
            "description": "Array of floats specifying the position. [x, y, z]"
        },
        "orientation": {
            "type": "array",
            "items": {
            "type": "number"
            },
            "description": "Array of floats specifying the orientation. [x, y, z, w]",
            "default": null
        },
        "relative": {
            "type": "boolean",
            "description": "Boolean to indicate if the move should be relative to the current position.",
            "default": false
        }
        },
        "required": [
        "position"
        ]
    }
    },

    {
    "name": "object_segmentation",
    "parameters": {
        "type": "object",
        "properties": {
        "image": {
            "type": "string",
            "description": "File path to the input image."
        },
        "prompt": {
            "type": "string",
            "description": "Description of the object to be segmented."
        }
        },
        "required": [
        "image",
        "prompt"
        ]
    },
    "description": "Segment an object in an image based on a given prompt."
    },

    {
    "name": "object_position",
    "parameters": {
        "type": "object",
        "properties": {
        "seg": {
            "type": "string",
            "description": "File path to the segmentation mask."
        },
        "depth": {
            "type": "string",
            "description": "File path to the depth image."
        }
        },
        "required": [
        "seg",
        "depth"
        ]
    },
    "description": "Calculate the object position in the robot coordinates."
    },

    {
    "name": "capture_image",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    },
    "description": "Capture an image and its corresponding depth map using the camera."
    },

    {
    "name": "view_image",
    "parameters": {
        "type": "object",
        "properties": {
        "file_path": {
            "type": "string",
            "description": "File path to the image. Put 'camera' if you would like to view a fresh image taken by the camera. You can view segmentation masks as well and they will be rendered as a black and white image."
        }
        },
        "required": []
    },
    "description": "View an image from a file or camera."
    },

    {
    "name": "get_robot_pose",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    },
    "description": "Get the current position and orientation of the robot."
    },

    {
    "name": "open_gripper",
    "description": "Open the robot gripper.",
    "parameters": {
        "type": "object",
        "properties": {}
    },
    "required": []
    },

    {
    "name": "close_gripper",
    "description": "Close the robot gripper.",
    "parameters": {
        "type": "object",
        "properties": {}
    },
    "required": []
    },
]



example_responses = '''[
    {rgb_path: "rgp.jpg", depth_path: "depth.npy"},
    {masks: ["mask_1.npy", "mask_2.npy"]},
    {
        object_center: [0.105, -0.287, 0.05],
        object_width: 0.14, object_height: 0.05, object_depth: 2.03,
        rotation_along_width: 0.152,
        rotation_along_height: 2.946,    
    },
    {
        object_center: [0.105, -0.287, 0.05],
        object_width: 0.03, object_height: 0.05, object_depth: 0.03,
        rotation_along_width: 0.152,
        rotation_along_height: 2.946,    
    },

]'''
