"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Joint Monkey
------------
- Animates degree-of-freedom ranges for a given asset.
- Demonstrates usage of DOF properties and states.
- Demonstrates line drawing utilities to visualize DOF frames (origin and axis).
"""

import math
import numpy as np
from isaacgym import gymapi, gymutil
from envs.motion_generator import MotionGenerator
from isaacgym import gymtorch
import torch
from IPython import embed


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

# simple asset descriptor for selecting from a list


# parse arguments
args = gymutil.parse_arguments(
    description="visualize input motion capture data",
    custom_parameters=[
        {"name": "--mocap", "type": str, "default": "walk.txt", "help": "mocap file"}])

# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)


# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 1.0, 0.0)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
asset_root = "./data/xml"
asset_file = "humanoid.xml"

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset_options = gymapi.AssetOptions()
asset_options.disable_gravity = True
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# get array of DOF names
dof_names = gym.get_asset_dof_names(asset)

# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset)

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# get the position slice of the DOF state array
dof_positions = dof_states['pos']

# get the limit-related slices of the DOF properties array
stiffnesses = dof_props['stiffness']
dampings = dof_props['damping']
armatures = dof_props['armature']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']

# initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
defaults = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)
for i in range(num_dofs):
    if has_limits[i]:
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
            upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
        # make sure our default position is in range
        if lower_limits[i] > 0.0:
            defaults[i] = lower_limits[i]
        elif upper_limits[i] < 0.0:
            defaults[i] = upper_limits[i]
    else:
        # set reasonable animation limits for unlimited joints
        if dof_types[i] == gymapi.DOF_ROTATION:
            # unlimited revolute joint
            lower_limits[i] = -math.pi
            upper_limits[i] = math.pi
        elif dof_types[i] == gymapi.DOF_TRANSLATION:
            # unlimited prismatic joint
            lower_limits[i] = -1.0
            upper_limits[i] = 1.0
    # set DOF position to default
    dof_positions[i] = defaults[i]
    # set speed depending on DOF type and range of motion

# Print DOF properties
for i in range(num_dofs):
    print("DOF %d" % i)
    print("  Name:     '%s'" % dof_names[i])
    print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
    print("  Stiffness:  %r" % stiffnesses[i])
    print("  Damping:  %r" % dampings[i])
    print("  Armature:  %r" % armatures[i])
    print("  Limited?  %r" % has_limits[i])
    if has_limits[i]:
        print("    Lower   %f" % lower_limits[i])
        print("    Upper   %f" % upper_limits[i])

# set up the env grid
num_envs = 1
num_per_row = 1
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(10.2, 2.0, 2)
cam_target = gymapi.Vec3(5, -2.5, 13)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
    pose.r = gymapi.Quat(1, 0.0, 0.0, 0.0)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

_root_tensor = gym.acquire_actor_root_state_tensor(sim)
root_tensor = gymtorch.wrap_tensor(_root_tensor)
motion_generator = MotionGenerator('./data/bvh/' + args.mocap)
t = 0
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    #gym.simulate(sim)
    gym.fetch_results(sim, True)

    pose,vel = motion_generator.generate_batch(np.array([t, t+0.2, t+0.4, t+0.6]))
    root_tensor[0,0:7] = torch.Tensor(pose[0, 0:7])
    root_tensor[0,7:13] = torch.Tensor(vel[0, 0:6])

    dof_states['pos'] = pose[0, 7:]
    dof_states['vel'] = vel[0, 6:]

   
    for i in range(num_envs):
        gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_tensor))

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    t += 1/120

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)