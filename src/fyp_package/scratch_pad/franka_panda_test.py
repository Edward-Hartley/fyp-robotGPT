# #%%
# import numpy as np
# #%%

# import pybullet as p
# import pybullet_data

# # if gui server not running, start one
# if p.getConnectionInfo()['isConnected'] == 0:
#     p.connect(p.GUI)

# # import urdf
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# plane = p.loadURDF("plane.urdf")
# panda = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
# print(p.getNumJoints(panda))
# p.setGravity(0, 0, -9.81)

# # show camera view
# p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0])

# #%%

# # get joint info
# num_joints = p.getNumJoints(panda)
# print(num_joints)
# for i in range(num_joints):
#     # format getJointInfo: (jointIndex, jointName, jointType, qIndex, uIndex, flags, jointDamping, jointFriction, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, linkName, jointAxis, parentFramePos, parentFrameOrn, parentIndex)
#     # pretty print joint info
#     print(p.getJointInfo(panda, i))
#     print(p.getLinkState(panda, i))
# print(p.getLinkState(panda, 11))


# # set joint positions
# p.setJointMotorControlArray(panda, range(num_joints), p.POSITION_CONTROL, targetPositions=[0.0] * num_joints)

# p.resetJointState(panda, 9, 0)

# home_joints = [-np.pi / 2,0.2,0,-1.3,0, 1.6, np.pi/4, 0, 0]


# # %%
# joint_ids = [p.getJointInfo(panda, i) for i in range(p.getNumJoints(panda))]
# revolute_joint_ids = [j[0] for j in joint_ids if j[2] == p.JOINT_REVOLUTE]
# fixed_joints = [j[0] for j in joint_ids if j[2] == p.JOINT_FIXED]
# print(fixed_joints)

# # Move robot to home configuration.
# for i in range(len(revolute_joint_ids)):
#     p.resetJointState(panda, revolute_joint_ids[i], home_joints[i])

# # set joint 7 to different colour
# p.changeVisualShape(panda, 6, rgbaColor=[1, 0, 0, 1])
# # %%
# p.resetJointState(panda, 10, 0.04)
# p.resetJointState(panda, 9, 0.04)

# lpad = np.array(p.getLinkState(panda, 9)[0])
# rpad = np.array(p.getLinkState(panda, 10)[0])
# dist = np.linalg.norm(lpad - rpad) - 0.02
# print(dist)

# # %%
# ee_pos = np.array(p.getLinkState(panda, 8)[0])
# tool_pos = np.array(p.getLinkState(panda, 11)[0])
# vec = (tool_pos - ee_pos) # / np.linalg.norm((tool_pos - ee_pos))
# ee_targ = ee_pos + vec
# ray_data = p.rayTest(ee_pos, ee_targ)[0]
# obj, link, ray_frac = ray_data[0], ray_data[1], ray_data[2]
# print( obj, link, ray_frac)

# object_id = p.loadURDF('./pybullet_assets/' + "bowl/bowl.urdf", [0.0, -0.5, 0.0], useFixedBase=1)

# # %%
# # get dimensions of object
# object_dimensions = p.getAABB(object_id)
# print(object_dimensions)
# object_dimensions = np.array(object_dimensions[1]) - np.array(object_dimensions[0])
# print(object_dimensions)

# # plane_id = p.loadURDF("plane.urdf", [0, 0, -0.001])
# # print(p.getAABB(plane_id))


# input("Press Enter to continue...")

from fyp_package.environment import environment
import pybullet

env = environment.SimulatedEnvironment(3, 3)
for _ in range(300):
    env.sim.step_sim_and_render()

for joint in [pybullet.getJointInfo(env.sim.robot_id, i) for i in range(pybullet.getNumJoints(env.sim.robot_id))]:
    print(joint)


print(env.sim.get_ee_pos()[2])
env.sim.gripper.release()
for _ in range(300):
    env.sim.step_sim_and_render()

env.sim.move_ee([-0.1, 0.4, -0.0])
for _ in range(300):
    env.sim.step_sim_and_render()

print(env.sim.get_ee_pos())

input("Press Enter to continue...")

