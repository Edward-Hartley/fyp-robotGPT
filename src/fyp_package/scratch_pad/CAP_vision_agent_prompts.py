prompt_tabletop_ui = '''
# Python 2D robot control script
import numpy as np
from env_utils import put_first_on_second, get_obj_names, say, get_corner_name, get_side_name, move_robot, move_robot_relative, open_gripper, close_gripper
from plan_utils import parse_obj_name, parse_position, parse_question, transform_shape_pts, vision_assistant

# Anything which can be picked up by grasping its center is easy to pick and place.
# Some objects are harder, like cups, bowls, pans, etc. which require grasping a specific part.
# For these objects we need to ask the vision assistant for the best place to grasp them.
# For fine control over movement you can also use the move_robot and move_robot_relative functions, they require 3D positions.
objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# place the yellow block on the yellow bowl.
say('Ok - putting the yellow block on the yellow bowl')
yellow_block_pos = vision_assistant('Return the [x, y] position of the yellow block')
yellow_bowl_pos = vision_assistant('Return the [x, y] position of the yellow bowl')
put_first_on_second(yellow_block_pos, yellow_bowl_pos)
objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# which block did you move.
say('I moved the yellow block')
objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# move the green block to the top right corner.
say('Got it - putting the green block on the top right corner')
corner_pos = parse_position('top right corner')
green_block_pos = vision_assistant('Return the [x, y] position of the green block')
put_first_on_second(green_block_pos, corner_pos)
objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# stack the blue bowl on the yellow bowl on the green block.
order_bottom_to_top = ['green block', 'yellow block', 'blue bowl']
say(f'Sure - stacking from top to bottom: {", ".join(order_bottom_to_top)}')
stack_objects_in_order(object_names=order_bottom_to_top)
objects = ['cyan block', 'white block', 'cyan bowl', 'blue block', 'blue bowl', 'white bowl']
# move the cyan block into its corresponding bowl.
matches = {'cyan block': 'cyan bowl'}
say('Got it - placing the cyan block on the cyan bowl')
for first, second in matches.items():
  put_first_on_second(vision_assistant(f'Return the [x, y] position of the {first}'), vision_assistant(f'Return the [x, y] position of the {second}'))
objects = ['cyan block', 'white block', 'cyan bowl', 'blue block', 'blue bowl', 'white bowl']
# make a line of blocks on the right side.
say('No problem! Making a line of blocks on the right side')
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
line_pts = parse_position(f'a 30cm vertical line on the right with {len(block_names)} points')
for block_name, pt in zip(block_names, line_pts):
  put_first_on_second(vision_assistant(f'Return the [x, y] position of the {block_name}'), pt)
objects = ['yellow block', 'red block', 'yellow bowl', 'gray block', 'gray bowl', 'red bowl']
# put the small banana colored thing in between the blue bowl and green block.
say('Sure thing - putting the yellow block between the blue bowl and the green block')
yellow_block_pos = vision_assistant('Return the [x, y] position of the yellow block')
target_pos = parse_position('a point in the middle betweeen the blue bowl and the green block')
put_first_on_second(yellow_block_pos, target_pos)
objects = ['yellow block', 'red block', 'yellow bowl', 'gray block', 'gray bowl', 'red bowl']
# can you cut the bowls in half.
say('no, I can only move objects around')
objects = ['yellow block', 'green block', 'yellow bowl', 'gray block', 'gray bowl', 'green bowl']
# stack the blocks on the right side with the gray one on the bottom.
say('Ok. stacking the blocks on the right side with the gray block on the bottom')
gray_block_pos = vision_assistant('Return the [x, y] position of the gray block')
right_side = parse_position('the right side')
put_first_on_second(gray_block_pos, right_side)
order_bottom_to_top = ['gray block', 'green block', 'yellow block']
stack_objects_in_order(object_names=order_bottom_to_top)
objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# hide the blue bowl.
bowl_name = np.random.choice(['yellow bowl', 'green bowl'])
say(f'Sounds good! Hiding the blue bowl under the {bowl_name}')
say('bowls are hard to grasp, so we need to ask the vision assistant for the best place to grasp them.')
bowl_pick_pos, bowl_pick_angle = vision_assistant(f'Return the [x, y, z] position and angle of best grasp for the {bowl_name}')
put_first_on_second(bowl_pick_pos, 'blue bowl', bowl_pick_angle)
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# place the top most block to the corner closest to the bottom most block.
top_block_name = parse_obj_name('top most block', f'objects = {get_obj_names()}')
bottom_block_name = parse_obj_name('bottom most block', f'objects = {get_obj_names()}')
closest_corner_pos = parse_position(f'the corner closest to the {bottom_block_name}', f'objects = {get_obj_names()}')
say(f'Putting the {top_block_name} on the {get_corner_name(closest_corner_pos)}')
put_first_on_second(vision_assistant(f'Return the [x, y] position of the {top_block_name}'), closest_corner_pos)
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# place the green block to the right of the bowl that has the blue block.
bowl_name = parse_obj_name('the bowl that has the blue block', f'objects = {get_obj_names()}')
if bowl_name:
  green_block_pos = vision_assistant('Return the [x, y] position of the green block')
  target_pos = parse_position(f'a point 10cm to the right of the {bowl_name}')
  say(f'No problem - placing the green block to the right of the {bowl_name}')
  put_first_on_second(green_block_pos, target_pos)
else:
  say('There are no bowls that has the blue block')
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# move the other blocks to the bottom corners.
block_names = parse_obj_name('blocks other than the blue block', f'objects = {get_obj_names()}')
corners = parse_position('the bottom corners')
for block_name, pos in zip(block_names, corners):
  put_first_on_second(vision_assistant(f'Return the [x, y] position of the {block_name}'), pos)
objects = ['yellow bowl', 'blue block', 'yellow block', 'blue bowl']
# is the blue block to the right of the yellow bowl?
if parse_question('is the blue block to the right of the yellow bowl?', f'objects = {get_obj_names()}'):
  say('yes, there is a blue block to the right of the yellow bow')
else:
  say('no, there is\'t a blue block to the right of the yellow bow')
objects = ['yellow bowl', 'blue block', 'yellow block', 'blue bowl']
# how many yellow objects are there?
n_yellow_objs = parse_question('how many yellow objects are there', f'objects = {get_obj_names()}')
say(f'there are {n_yellow_objs} yellow object')
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# what are the blocks left of the green bowl.
left_block_names =  parse_question('what are the blocks left of the green bowl', f'objects = {get_obj_names()}')
if len(left_block_names) > 0:
  say(f'These blocks are left of the green bowl: {", ".join(left_block_names)}')
else:
  say('There are no blocks left of the green bowl')
objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# imagine that the bowls are different biomes on earth and imagine that the blocks are parts of a building.
say('ok')
objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# now build a tower in the grasslands.
order_bottom_to_top = ['green bowl', 'blue block', 'green block', 'yellow block']
say('stacking the blocks on the green bowl')
stack_objects_in_order(object_names=order_bottom_to_top)
objects = ['yellow block', 'green block', 'yellow bowl', 'gray block', 'gray bowl', 'green bowl']
# push the yellow block to the right 5cm.
say('Sure - pushing the yellow block to the right 5cm')
say('I will need finer control and will use 3D positions for this.')
yellow_block_pos_3d = vision_assistant('Return the [x, y, z] position of the yellow block')
close_gripper()
move_robot(yellow_block_pos_3d + [-0.03, 0, 0.1])
move_robot_relative([0, 0, -0.1])
move_robot_relative([0.05, 0, 0])
# move all blocks 5cm toward the top.
say('Ok - moving all blocks 5cm toward the top')
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
for block_name in block_names:
  target_pos = parse_position(f'a point 5cm above the {block_name}')
  put_first_on_second(vision_assistant(f'Return the [x, y] position of the {block_name}'), target_pos)
objects = ['cyan block', 'white block', 'purple bowl', 'blue block', 'blue bowl', 'white bowl']
# make a triangle of blocks in the middle.
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
triangle_pts = parse_position(f'a triangle with size 10cm around the middle with {len(block_names)} points')
say('Making a triangle of blocks around the middle of the workspace')
for block_name, pt in zip(block_names, triangle_pts):
  put_first_on_second(vision_assistant(f'Return the [x, y] position of the {block_name}'), pt)
objects = ['cyan block', 'white block', 'purple bowl', 'blue block', 'blue bowl', 'white bowl']
# make the triangle smaller.
triangle_pts = transform_shape_pts('scale it by 0.5x', shape_pts=triangle_pts)
say('Making the triangle smaller')
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
for block_name, pt in zip(block_names, triangle_pts):
  put_first_on_second(vision_assistant(f'Return the [x, y] position of the {block_name}'), pt)
objects = ['paper cup', 'red bowl']
# put the paper cup to the left of the bowl.
say('Sure - putting the paper cup to the left of the red bowl')
say('cups are hard to grasp, so we need to ask the vision assistant for the best place to grasp them.')
cup_pick_pos, cup_pick_angle = vision_assistant('Return the [x, y, z] position and angle of the best grasp point for the paper cup')
place_pos = parse_position('a point 10cm to the left of the red bowl')
put_first_on_second(cup_pick_pos, place_pos, pick_angle=cup_pick_angle)
objects = ['brown bowl', 'red block', 'brown block', 'red bowl', 'pink bowl', 'pink block']
# move the red bowl to the top right corner.
say('Got it - moving the red bowl to the top right corner')
say('bowls are hard to grasp, so we need to ask the vision assistant for the best place to grasp them.')
red_bowl_pick_pos, red_bowl_pick_angle = vision_assistant('Return the [x, y, z] position and angle of the red bowl')
top_right_corner_pos = parse_position('top right corner')
put_first_on_second(red_bowl_pick_pos, top_right_corner_pos, pick_angle=red_bowl_pick_angle)
objects = ['paper cup', 'paper cup', 'white bowl', 'red block']
# Put one cup inside the other.
say('Sure - putting one paper cup inside the other')
cup_positions = vision_assistant('Return a list of [x, y, z] positions of the paper cups')
pick_cup_pos = cup_positions[0]
place_cup_pos = cup_positions[1]
say('cups are hard to grasp, so we need to ask the vision assistant for the best place to grasp the one we are picking up.')
pick_cup_pick_pos, pick_cup_pick_angle = vision_assistant(f'Return the [x, y, z] position and angle of the best grasp point for the paper cup at {pick_cup_pos}')
put_first_on_second(pick_cup_pick_pos, place_cup_pos, pick_angle=pick_cup_pick_angle)
objects = ['brown bowl', 'red block', 'brown block', 'red bowl', 'pink bowl', 'pink block']
# rotate the red block 90 degrees
say('Sure - rotating the red block 90 degrees')
red_block_pos = vision_assistant('Return the [x, y] position of the red block')
put_first_on_second(red_block_pos, red_block_pos, pick_angle=np.pi/2)
'''.strip()

# %%
prompt_parse_obj_name = '''
import numpy as np
from env_utils import parse_position
from plan_utils import vision_assistant
from utils import get_obj_positions_np

objects = ['blue block', 'cyan block', 'purple bowl', 'gray bowl', 'brown bowl', 'pink block', 'purple block']
# the block closest to the purple bowl.
block_names = ['blue block', 'cyan block', 'purple block']
block_positions = get_obj_positions_np(block_names)
closest_block_idx = get_closest_idx(points=block_positions, point=vision_assistant('Return the [x, y] position of the purple bowl'))
closest_block_name = block_names[closest_block_idx]
ret_val = closest_block_name
objects = ['brown bowl', 'banana', 'brown block', 'apple', 'blue bowl', 'blue block']
# the blocks.
ret_val = ['brown block', 'blue block']
objects = ['brown bowl', 'banana', 'brown block', 'apple', 'blue bowl', 'blue block']
# the brown objects.
ret_val = ['brown bowl', 'brown block']
objects = ['brown bowl', 'banana', 'brown block', 'apple', 'blue bowl', 'blue block']
# a fruit that's not the apple
fruit_names = ['banana', 'apple']
for fruit_name in fruit_names:
    if fruit_name != 'apple':
        ret_val = fruit_name
objects = ['blue block', 'cyan block', 'purple bowl', 'brown bowl', 'purple block']
# blocks above the brown bowl.
block_names = ['blue block', 'cyan block', 'purple block']
brown_bowl_pos = vision_assistant('Return the [x, y] position of the brown bowl')
use_block_names = []
for block_name in block_names:
    if vision_assistant(f'Return the [x, y] position of the {block_name}')[1] > brown_bowl_pos[1]:
        use_block_names.append(block_name)
ret_val = use_block_names
objects = ['blue block', 'cyan block', 'purple bowl', 'brown bowl', 'purple block']
# the blue block.
ret_val = 'blue block'
objects = ['blue block', 'cyan block', 'purple bowl', 'brown bowl', 'purple block']
# the block closest to the bottom right corner.
corner_pos = parse_position('bottom right corner')
block_names = ['blue block', 'cyan block', 'purple block']
block_positions = get_obj_positions_np(block_names)
closest_block_idx = get_closest_idx(points=block_positions, point=corner_pos)
closest_block_name = block_names[closest_block_idx]
ret_val = closest_block_name
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# the left most block.
block_names = ['green block', 'brown block', 'blue block']
block_positions = get_obj_positions_np(block_names)
left_block_idx = np.argsort(block_positions[:, 0])[0]
left_block_name = block_names[left_block_idx]
ret_val = left_block_name
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# the bowl on near the top.
bowl_names = ['brown bowl', 'green bowl', 'blue bowl']
bowl_positions = get_obj_positions_np(bowl_names)
top_bowl_idx = np.argsort(bowl_positions[:, 1])[-1]
top_bowl_name = bowl_names[top_bowl_idx]
ret_val = top_bowl_name
objects = ['yellow bowl', 'purple block', 'yellow block', 'purple bowl', 'pink bowl', 'pink block']
# the third bowl from the right.
bowl_names = ['yellow bowl', 'purple bowl', 'pink bowl']
bowl_positions = get_obj_positions_np(bowl_names)
bowl_idx = np.argsort(bowl_positions[:, 0])[-3]
bowl_name = bowl_names[bowl_idx]
ret_val = bowl_name
'''.strip()

# %%
prompt_parse_position = '''
import numpy as np
from shapely.geometry import *
from shapely.affinity import *
from env_utils import denormalize_xy, parse_obj_name, get_obj_names
from plan_utils import vision_assistant

# a 30cm horizontal line in the middle with 3 points.
middle_pos = denormalize_xy([0.5, 0.5])
start_pos = middle_pos + [-0.3/2, 0]
end_pos = middle_pos + [0.3/2, 0]
line = make_line(start=start_pos, end=end_pos)
points = interpolate_pts_on_line(line=line, n=3)
ret_val = points
# a 20cm vertical line near the right with 4 points.
middle_pos = denormalize_xy([1, 0.5])
start_pos = middle_pos + [0, -0.2/2]
end_pos = middle_pos + [0, 0.2/2]
line = make_line(start=start_pos, end=end_pos)
points = interpolate_pts_on_line(line=line, n=4)
ret_val = points
# a diagonal line from the top left to the bottom right corner with 5 points.
top_left_corner = denormalize_xy([0, 1])
bottom_right_corner = denormalize_xy([1, 0])
line = make_line(start=top_left_corner, end=bottom_right_corner)
points = interpolate_pts_on_line(line=line, n=5)
ret_val = points
# a triangle with size 10cm with 3 points.
polygon = make_triangle(size=0.1, center=denormalize_xy([0.5, 0.5]))
points = get_points_from_polygon(polygon)
ret_val = points
# the corner closest to the sun colored block.
block_name = parse_obj_name('the sun colored block', f'objects = {get_obj_names()}')
corner_positions = np.array([denormalize_xy(pos) for pos in [[0, 0], [0, 1], [1, 1], [1, 0]]])
closest_corner_pos = get_closest_point(points=corner_positions, point=vision_assistant(f'Return the [x, y] position of the {block_name}'))
ret_val = closest_corner_pos
# the side farthest from the right most bowl.
bowl_name = parse_obj_name('the right most bowl', f'objects = {get_obj_names()}')
side_positions = np.array([denormalize_xy(pos) for pos in [[0.5, 0], [0.5, 1], [1, 0.5], [0, 0.5]]])
farthest_side_pos = get_farthest_point(points=side_positions, point=vision_assistant(f'Return the [x, y] position of the {bowl_name}'))
ret_val = farthest_side_pos
# a point above the third block from the bottom.
block_name = parse_obj_name('the third block from the bottom', f'objects = {get_obj_names()}')
ret_val = vision_assistant(f'Return the [x, y] position of the {block_name}') + [0.1, 0]
# a point 10cm left of the bowls.
bowl_names = parse_obj_name('the bowls', f'objects = {get_obj_names()}')
bowl_positions = get_all_object_positions_np(obj_names=bowl_names)
left_obj_pos = bowl_positions[np.argmin(bowl_positions[:, 0])] + [-0.1, 0]
ret_val = left_obj_pos
# the bottom side.
bottom_pos = denormalize_xy([0.5, 0])
ret_val = bottom_pos
# the top corners.
top_left_pos = denormalize_xy([0, 1])
top_right_pos = denormalize_xy([1, 1])
ret_val = [top_left_pos, top_right_pos]
'''.strip()

# %%
prompt_parse_question = '''
from utils import get_obj_names, parse_obj_name, bbox_contains_pt, vision_assistant

objects = ['yellow bowl', 'blue block', 'yellow block', 'blue bowl', 'fruit', 'green block', 'black bowl']
# is the blue block to the right of the yellow bowl?
ret_val = vision_assistant('Return the [x, y] position of the blue block')[0] > vision_assistant('Return the [x, y] position of the yellow bowl')[0]
objects = ['yellow bowl', 'blue block', 'yellow block', 'blue bowl', 'fruit', 'green block', 'black bowl']
# how many yellow objects are there?
yellow_object_names = parse_obj_name('the yellow objects', f'objects = {get_obj_names()}')
ret_val = len(yellow_object_names)
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# is the pink block on the green bowl?
ret_val = bbox_contains_pt(container_name='green bowl', obj_name='pink block')
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# what are the blocks left of the green bowl?
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
green_bowl_pos = vision_assistant('Return the [x, y] position of the green bowl')
left_block_names = []
for block_name in block_names:
  if vision_assistant(f'Return the [x, y] position of the {block_name}')[0] < green_bowl_pos[0]:
    left_block_names.append(block_name)
ret_val = left_block_names
objects = ['pink block', 'yellow block', 'pink bowl', 'blue block', 'blue bowl', 'yellow bowl']
# is the sun colored block above the blue bowl?
sun_block_name = parse_obj_name('sun colored block', f'objects = {get_obj_names()}')
sun_block_pos = vision_assistant(f'Return the [x, y] position of the {sun_block_name}')
blue_bowl_pos = vision_assistant('Return the [x, y] position of the blue bowl')
ret_val = sun_block_pos[1] > blue_bowl_pos[1]
objects = ['pink block', 'yellow block', 'pink bowl', 'blue block', 'blue bowl', 'yellow bowl']
# is the green block below the blue bowl?
ret_val = vision_assistant('Return the [x, y] position of the green block')[1] < vision_assistant('Return the [x, y] position of the blue bowl')[1]
'''.strip()

# %%
prompt_transform_shape_pts = '''
import numpy as np
from utils import get_obj_names, parse_position, parse_obj_name, vision_assistant

# make it bigger by 1.5.
new_shape_pts = scale_pts_around_centroid_np(shape_pts, scale_x=1.5, scale_y=1.5)
# move it to the right by 10cm.
new_shape_pts = translate_pts_np(shape_pts, delta=[0.1, 0])
# move it to the top by 20cm.
new_shape_pts = translate_pts_np(shape_pts, delta=[0, 0.2])
# rotate it clockwise by 40 degrees.
new_shape_pts = rotate_pts_around_centroid_np(shape_pts, angle=-np.deg2rad(40))
# rotate by 30 degrees and make it slightly smaller
new_shape_pts = rotate_pts_around_centroid_np(shape_pts, angle=np.deg2rad(30))
new_shape_pts = scale_pts_around_centroid_np(new_shape_pts, scale_x=0.7, scale_y=0.7)
# move it toward the blue block.
block_name = parse_obj_name('the blue block', f'objects = {get_obj_names()}')
block_pos = vision_assistant(f'Return the [x, y] position of the {block_name}')
mean_delta = np.mean(block_pos - shape_pts, axis=1)
new_shape_pts = translate_pts_np(shape_pts, mean_delta)
'''.strip()

# %%
prompt_fgen = '''
import numpy as np
from shapely.geometry import *
from shapely.affinity import *

from env_utils import get_obj_names
from plan_utils import vision_assistant
from ctrl_utils import put_first_on_second

# define function: total = get_total(xs=numbers).
def get_total(xs):
    return np.sum(xs)

# define function: y = eval_line(x, slope, y_intercept=0).
def eval_line(x, slope, y_intercept):
    return x * slope + y_intercept

# define function: pt = get_pt_to_the_left(pt, dist).
def get_pt_to_the_left(pt, dist):
    return pt + [-dist, 0]

# define function: pt = get_pt_to_the_top(pt, dist).
def get_pt_to_the_top(pt, dist):
    return pt + [0, dist]

# define function line = make_line_by_length(length=x).
def make_line_by_length(length):
  line = LineString([[0, 0], [length, 0]])
  return line

# define function: line = make_vertical_line_by_length(length=x).
def make_vertical_line_by_length(length):
  line = make_line_by_length(length)
  vertical_line = rotate(line, 90)
  return vertical_line

# define function: pt = interpolate_line(line, t=0.5).
def interpolate_line(line, t):
  pt = line.interpolate(t, normalized=True)
  return np.array(pt.coords[0])

# example: scale a line by 2.
line = make_line_by_length(1)
new_shape = scale(line, xfact=2, yfact=2)

# example: put object1 on top of object0.
object1_pos = vision_assistant('Return the [x, y] position of the object1')
object0_pos = vision_assistant('Return the [x, y] position of the object0')
put_first_on_second(object1_pos, object0_pos)

# example: get the position of the first object.
obj_names = get_obj_names()
pos_2d = vision_assistant(f'Return a list of the [x, y] positions of the {obj_names}')[0]
'''.strip()



# Top system prompt begins with system message
# All others begin with user message and then alternative assistant and system messages

vision_top_system_message = '''
You are an AI agent responsible for vision in a tabletop environment.You have tools that you can invoke via python code.
Follow the format below; code following a **CODE** tag should be python code, and the output following a **RET** tag should be the return value either as a variable or an explicit value.

All coordinates and dimensions are in meters. Before using any tools, you should outline reasonable expectations about the objects you have been tasked with detecting. When verifying results, use code to filter out unreasonable detections.
When models return unexpected results (multiple detections or zero detections, strange object dimensions), you should try using them with different prompts or using common sense to infer the correct detection (e.g. a banana should not have a width of 10 meters, or 'paper' might also be detected with the prompt 'piece of paper' or 'white paper'). Err on the side of caution.
Features that you can use to discriminate between objects include the absolute values of the dimensions, the relative values of the dimensions, and the relative positions of the objects.
When this does not work, you can try viewing an image of the environment yourself to gain contextual information. You can only use this information to verify results and guide future model uses. You may use code to create your own images to view, and then use display_image to view them.
You can only view a single image at a time.

On the tabletop system, these functions are already defined:
{functions_docs}
You also have access to the following packages:
{packages}

The following messages are example histories of interactions with the environment. The environment resets upon each user query.
Take special care to ensure the returned value is wrapped in the correct way - if a list is expected, return a list, if a single value is expected, return a single value.
'''.strip()

# When models return unexpected results, you can use the `display_image` function to view the images, masks or other visual data that you have constructed for vision verification.

vision_get_images_example = [
'''
Please get a camera image from the environment and save it in a file.
Return the name of the new file.
'''.strip(),

'''
Expectations:
* There may be existing files in the directory.
* There should be no errors thrown at any point.
* No verification functions are required, I will continue with the task
I first check what files exist to make sure I don't overwrite anything.
For this I can use the `get_images` method of the environment. I will then save the image to a file using the `cv2.imwrite` method.
**CODE**
files = os.listdir('.')
print(files)
'''.strip(),

'''
stdout:
['camera_image_0.jpg', 'camera_image_1.jpg']
'''.strip(),

'''
I will now get the camera image and try to save it to a file.
**CODE**
rgb, depth = get_images()
cv2.imwrite('camera_image_2.jpg')
'''.strip(),

'''
stdout: 
'''.strip(),

'''
No errors were output, I successfully saved the camera image to a file called `camera_image_2.jpg`.
**RET**
camera_image_2.jpg
'''.strip(),
]

vision_generic_example = [
'''
[User query specifying request and output format]
'''.strip(),

'''
Expecations:
[* list of expectations]
[Reasoning about the user query]
**CODE**
[Small snippet of code]
[print statements showing intermediate results if needed]
'''.strip(),

'''
stdout:
[Printed results of the code snippet, some results are unexpected]
'''.strip(),

'''
[Reasoning about results and next steps to make progress/interpret results]
**CODE**
[Small snippet of code]
[print statements showing further results]
'''.strip(),

'''
stdout:
[Printed results of the code snippet]
'''.strip(),

'''
[Final reasoning about results and next steps, compare results against expectations]
**RET**
[Return values - variable or explicit value, matching the user query]
'''.strip(),
]

vision_detect_object_example = [
'''
Please give me the [x, y, z] positions of all visible bananas as a list of lists.
'''.strip(),

'''
Expectations:
* There may be one or multiple bananas in the scene.
* The detections may not be perfect, and some post-processing may be required.
* Bananas are typically long and narrow, with no dimension over 0.25 meters.
I will use the `detect_object` method to get the positions of all bananas in the scene.
I will view the detections first before deciding which ones to return.
**CODE**
rgb, depth = get_images()
detections, masks = detect_object('banana', rgb, depth)
'''.strip(),

'''
stdout:
Detection 1
Position of banana: [-0.166, -0.616, 0.048]
Dimensions:
Width: 0.048
Length: 0.153
Height: 0.048
Orientation along shorter side (width): 0.03
Orientation along longer side (length): -1.541

Detection 2
Position of banana: [-0.001, -0.505, 0.049]
Dimensions:
Width: 1.201
Length: 1.187
Height: 0.263
Orientation along shorter side (width): 1.568
Orientation along longer side (length): -0.002  

Detection 3
Position of banana: [0.207, -0.416, 0.048]
Dimensions:
Width: 0.051
Length: 0.143
Height: 0.042
Orientation along shorter side (width): 1.321
Orientation along longer side (length): -0.201

Total number of detections made: 3
'''.strip(),

'''
I have three detections, but the detection model can make mistakes.
I can check: the absolute values of the dimensions and the relative values of the dimensions.
**CODE**
filtered_detections = []
for detection in detections:
    # Bananas are typically long and narrow, with no dimension over 0.25 meters.
    if detection['width'] < 0.25 and detection['length'] < 0.25:
        if abs(detection['width'] - detection['length']) >= 0.05:
            filtered_detections.append(detection)
for detection in filtered_detections:
    print(f"Dimensions: {detection['width']}, {detection['length']}, {detection['height']}")
'''.strip(),

'''
stdout:
Dimensions: 0.048, 0.153, 0.048
Dimensions: 0.051, 0.143, 0.042
'''.strip(),

'''
I have filtered out the detections based on the dimensions and the relative values of the dimensions.
The remaining detections are reasonable and can be returned.
The user wanted a list of positions where each position is a list of points.
**RET**
[detection['position'] for detection in filtered_detections]
'''.strip(),
]

vision_detect_grasp_example = [
'''
Please give me the [x, y, z] position and the angle of the best grasp for the paper cup.
'''.strip(),

'''
Expectations:
* There is probably only a single paper cup.
* Paper cups are typically small, with dimensions less than 0.15 meters.
* The paper cup may on it's side or upright, but either way one of the dimensions should be slightly more than the other two.
* The grasp position should be on part of the object.
First I must detect the paper cup.
I will use 'detect_object' to get potential positions and masks and then review the results.
**CODE**
rgb, depth = get_images()
detections, masks = detect_object('paper cup', rgb, depth)
'''.strip(),

'''
stdout:
Detection 1
Position of paper cup: [0.191, -0.140, 0.107]
Dimensions:
Width: 0.133
Length: 0.086
Height: 0.007
Orientation along shorter side (width): -1.048
Orientation along longer side (length): -0.501

Detection 2
Position of paper cup: [-0.063, -0.320, 0.118]
Dimensions:
Width: 0.086
Length: 0.123
Height: 0.070
Orientation along shorter side (width): 0.743
Orientation along longer side (length): -0.892

Total number of detections made: 2
'''.strip(),

'''
I have two detections, but the detection model can make mistakes.
I can check: the absolute values of the dimensions and the relative values of the dimensions.
Both meet my prior expectations about paper cups, however the first detection has an abnormally small height.
I will check my interpretation using code.
I also need to get the corresponding masks for grasp detection.
**CODE**
filtered_detections = []
filtered_masks = []
for (detection, mask) in zip(detections, masks):
    # Paper cups are typically small, with one dimension slightly more than the other two.
    if detection['width'] < 0.15 and detection['length'] < 0.15 and detection['height'] < 0.15 and detection['height'] > 0.04:
        # The largest dimension should be significantly bigger and the other two should be similar.
        # This could be any of the dimensions as a paper cup can be on its side or upright.
        dimensions = [detection['width'], detection['length'], detection['height']]
        dimensions.sort(reverse=True)
        if dimensions[0] - dimensions[1] >= 0.03 and dimensions[1] - dimensions[2] <= 0.03:
            filtered_detections.append(detection)
            filtered_masks.append(mask)
for detection in filtered_detections:
    print(f"Dimensions: {detection['width']}, {detection['length']}, {detection['height']}")
print("Number of filtered detections:", len(filtered_detections))
'''.strip(),

'''
stdout:
Dimensions: 0.086, 0.123, 0.070
Number of filtered detections: 1
'''.strip(),
  
'''
I have filtered out the detections based on the dimensions and the relative values of the dimensions.
I now have a single detection that meets my expectations.
I have a list, but pass a single element to the grasp detection function as it does not take a list.
I can now proceed with the grasp detection.
**CODE**
grasp_position = detect_grasp(filtered_masks[0], depth)
'''.strip(),

'''
stdout:
No grasp detected. Returned None, None.
'''.strip(),

'''
The grasp detection did not return a grasp position, this could be due to the orientation of the object or the quality of the mask.
I want to check whether this mask correctly points to a paper cup.
I will use the `display_image` function to view the mask and verify that it is correct.
When viewing an image I must remind myself:
* I can only view a single image at a time.
* I should only use this information to verify results and guide future model uses.
* I can only infer simple information such as the existence of objects or their rough relative positions.
**CODE**
display_image(filtered_masks[0])
'''.strip(),

'''
stdout:
Image redacted in example. It shows a black and white mask.
'''

'''
The image of just the mask is hard to interpret, the black and white shape is very abstract.
I will create my own image to view the mask in context.
**CODE**
# overlay the mask on the rgb image as 
overlay = np.zeros_like(rgb)
overlay[filtered_masks[0]] = [255, 0, 0]
rgb_overlay = cv2.addWeighted(rgb, 0.5, overlay, 0.5, 0)
display_image(rgb_overlay)
'''

'''
stdout:
Image redacted in example. It shows the rgb image with the mask overlayed, highlighted in red.
'''

'''
I can see that the mask successfully captures the paper cup. However, the paper cup in the image is on it's side, which may be causing the grasp detection to fail.
Also, I am allowed to reason about the existence of objects in the image, and there was only one paper cup. I will not try to detect a grasp using the other mask.
The grasp detection model may be unreliable for objects on their side, I am going to try it again before considering alternatives.
**CODE**
grasp_position, grasp_angle = detect_grasp(filtered_masks[0], depth)
'''.strip(),

'''
stdout:
Detected grasp with contact point at [-0.053, -0.329, 0.120]
Grasp rotation about z-axis: 0.239
'''.strip(),

'''
I have successfully detected a grasp position for the paper cup.
The user wanted the grasp position and angle, so I will return these values.
**RET**
grasp_position, grasp_angle
'''.strip(),
]

vision_final_system_message = '''
The following message is the real user's query. All previous variables are no longer defined and you must start from scratch.
'''.strip()

vision_function_docs = {
    "get_images": '''
    Get an rgb and depth image from the camera.

    Returns:
        rgb: a numpy array representing the rgb image, shape (height, width, 3)
        depth: a numpy array representing the depth image, shape (height, width)

    Example:
        rgb, depth = get_images() # np.uint8[image_height, image_width, 3], np.float32[image_height, image_width]
    ''',
    "detect_object": '''
    Get object detections and corresponding masks from the rgb image and depth array.
    It will neatly print the results of the detection and segmentation as well as how many detections were made.
    It will return the detections and masks for further processing.
    Make sure that you have used the `get_images` function to define rgb and depth before using this function.

    Args:
        object_name: a string representing the object to detect
        rgb: a previously-defined numpy array representing the rgb image, shape (height, width, 3)
        depth: a previously-defined numpy array representing the depth image, shape (height, width)

    Returns:
        detections: a list of dictionaries containing the detections, each dictionary contains the keys 'position', 'width', 'length', 'height'
        masks: a list of numpy arrays of bools representing the masks of the detected objects, shape (height, width)

    Example:
        detections, masks = detect_object('banana', rgb, depth)
        object_1_position = detections[0]['position'] # [x, y, z]
        object_1_width = detections[0]['width'] # float
        object_1_length = detections[0]['length']
        object_1_height = detections[0]['height']
        object_1_mask = masks[0] # bool[image_height, image_width]
    ''',
    "display_image": '''
    Request to view an image yourself, it will be added in the following message.
    You should only use this to view rgb images, masks or your own images that you have constructed for vision verification.

    Args:
        image_array: a numpy array representing an image, shape (height, width, 3) or shape (height, width)

    Example:
        display_image(rgb)
        display_image(mask)
        display_image(my_image_array)
    ''',
    "detect_grasp": '''
    Get a suggested position of the best grasp available for an object from its segmentation mask and depth array.
    It will neatly print the results of the grasp detection.
    Make sure that you have a depth array and have selected the best object mask before using this function.

    Args:
        mask: a numpy array of bools representing the mask of the object, shape (height, width)
        depth: a numpy array representing the depth image, shape (height, width)

    Returns:
        grasp_position: a single list of points, [x, y, z] representing the suggested grasp position
        grasp_angle: a single float representing the suggested grasp angle around the z-axis

    Example:
        grasp_position, grasp_angle = detect_grasp(mask, depth)
    ''',
}

vision_confirm_return = '''
This is a standard check to ensure that the return value is correct. It may already be correct.
The returned string evaluated to {ret_val}.
The original user query was "{user_query}".
If the returned value is correct please return it again using **RET**.
If you would like to modify the format of the returned value please do so and return it using **RET**.
Do not modify the actual values of the returned variables, only the format.
If a single position was requested but a list of positions was returned, choose the first position and return it in the correct format.
'''.strip()

example_first_completion = '''
Let's start by detecting the espresso cup.
**CODE**
rgb, depth = get_images()
detections, masks = detect_object('espresso cup',rgb, depth)
'''

# GPT can only return either a tool usage or a completion, so we use our own tools instead for now
code_tool = {
    'type': 'function',
    'function': {
        'description': 'Execute the given code on the tabletop system, stdout will be returned to you.',
        'name': 'tabletop_execute',
        'parameters': {
            'type': 'object',
            'properties': {
                'code': {
                    'type': 'string',
                    'description': 'The code to execute on the tabletop system.'
                }
            },
            'required': ['code']
        }
    }
}
return_tool = {
    'type': 'function',
    'function': {
        'description': 'Return the given value to the user.',
        'name': 'tabletop_return',
        'parameters': {
            'type': 'object',
            'properties': {
                'value': {
                    'type': 'string',
                    'description': 'The value to return to the user.'
                }
            },
            'required': ['value']
        }
    }
}

example_detect_bowl = '''\nDetection 1\nPosition of bowl: [-0.169, -0.315, 0.042]\nDimensions:\nWidth: 0.172\nLength: 0.212\nHeight: 0.07\nOrientation along shorter side (width): 0.436\nOrientation along longer side (length): -1.135 \n\nDetection 2\nPosition of bowl: [0.01, -0.294, 0.042]\nDimensions:\nWidth: 0.099\nLength: 0.06\nHeight: 0.069\nOrientation along shorter side (length): -0.077\nOrientation along longer side (width): 1.494 \n\nDetection 3\nPosition of bowl: [-0.214, -0.171, 0.002]\nDimensions:\nWidth: 0.086\nLength: 0.101\nHeight: 0.027\nOrientation along shorter side (width): 0.07\nOrientation along longer side (length): -1.501 \n\nTotal number of detections made: 3'''
