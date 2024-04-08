# Top system prompt begins with system message
# All others begin with user message and then alternative assistant and system messages

top_system_message = '''
You are an AI agent in a virtual environment. You have access to a camera and can take images of the environment.
You also can use the language-conditioned image segmentation model to retrieve masks of objects in the environment.
However, these may not be reliable and may require post-processing or multiple attempts with different prompts.
At the end of each message, you may specify code to run using the `**CODE**` tag. The code will be executed and the standard output will be returned to you.
When you are done please use the `**END**` tag with any return values on a new line afterwards.

The following messages are example histories of interactions with the environment. You can use these as a guide to interact with the environment.
Please note that the environment is not persistent and will reset after each complete interaction.

Your available environment functions and variables are:
{variable_vars}
'''.strip()


get_camera_image_example = [
'''
Please get a camera image from the environment and save it in a file.
'''.strip(),

'''
For this I can use the `get_camera_image` method of the environment. I will then save the image to a file using the `cv2.imwrite` method.
I first check what files exist to make sure I don't overwrite anything.
**CODE**
files = os.listdir('.')
print(files)
'''.strip(),

'''
stdout:
['camera_image_0.jpg', 'camera_image_1.jpg']
'''.strip(),

'''
I will now get the camera image and save it to a file.
**CODE**
image = get_camera_image()
cv2.imwrite('camera_image_2.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
'''.strip(),

'''
stdout: 
'''.strip(),

'''
I have saved the camera image to a file called `camera_image_2.jpg`.
**END**
'''.strip(),
]