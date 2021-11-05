# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from posixpath import join
import sys
import os

from envs.motion_generator import MotionGenerator
sys.path.append('/home/flood/Documents/isaac_deepmimic/')

from utils.torch_jit_utils import *
from .base_env import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi


import numpy as np
import os
import torch

class Humanoid(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        #setting simulation parameters
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        #first we ignore these settings
        # self.randomization_params = self.cfg["task"]["randomization_params"]
        # self.randomize = self.cfg["task"]["randomize"]
        # self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        # self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        # self.power_scale = self.cfg["env"]["powerScale"]
        # self.heading_weight = self.cfg["env"]["headingWeight"]
        # self.up_weight = self.cfg["env"]["upWeight"]
        # self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        # self.energy_cost_scale = self.cfg["env"]["energyCost"]
        # self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        # self.death_cost = self.cfg["env"]["deathCost"]
        # self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.t = np.zeros((self.cfg["env"]["numEnvs"], )) #the number of simulation steps per environment
        self.num_steps = np.zeros((self.cfg["env"]["numEnvs"], ))

        #set these important parameters
        self.cfg["env"]["numObservations"] = 82
        self.cfg["env"]["numActions"] = 55
        
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 15.0, 2.4)
            cam_target = gymapi.Vec3(0.0, 2.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
       

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rb_states = gymtorch.wrap_tensor(rb_states)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.rb_pos = self.rb_states.view(self.num_envs, self.num_bodies, 13)[:, :, 0:3]
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        # initialize some data used later on
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()#facing vector
        self.basis_vec1 = self.up_vec.clone() #up vector
        self.dt = 1/60

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'y')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 1.0, 0.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.dirname(__file__) + '/../data'
        asset_file = "xml/humanoid.xml"
        task_motion = "/bvh/" + self.cfg['env']['asset']['task'] + '.txt'

        #initialize the kinematic motion planner
        self.motion_generator = MotionGenerator(asset_root + task_motion)

        # if "asset" in self.cfg["env"]:
        #     asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
        #     asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.angular_damping = 0.0
        asset_options.disable_gravity = False
      
        #
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        

        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        print('num_bodies:{}'.format(self.num_bodies))
        print('num_dof:{}'.format(self.num_dof))
        #we might need to change this settings for control
        #actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
    
        #motor_efforts = [prop.motor_effort for prop in actuator_props]
        #self.joint_gears = to_torch(motor_efforts, device=self.device)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0, 1.2, 0)
        start_pose.r = gymapi.Quat(0, 0, 0, 1)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        body_names = [self.gym.get_asset_rigid_body_name(humanoid_asset, i) for i in range(self.num_bodies)]
        extremity_names = [s for s in body_names if "foot" in s] # edn effector
        print('extremity_names:{}'.format(extremity_names))
        self.extremities_index = torch.zeros(len(extremity_names), dtype=torch.long, device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid" + str(i), i, 1, 0)

            env_sensors = []
            for extr in extremity_names:
                extr_handle = self.gym.find_actor_rigid_body_handle(env_ptr, humanoid_handle, extr)
                env_sensors.append(self.gym.create_force_sensor(env_ptr, extr_handle, sensor_pose))
                self.sensors.append(env_sensors)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            body_names = self.gym.get_actor_rigid_body_names(env_ptr, humanoid_handle)
            print(body_names)
            joint_names = self.gym.get_actor_joint_names(env_ptr, humanoid_handle)
            print(joint_names)
            joint_dict = self.gym.get_actor_joint_dict(env_ptr, humanoid_handle)
            print(" ")

        
            self.envs.append(env_ptr)
            self.humanoid_handles.append(humanoid_handle)

            #set control mode and parameters
            dof_prop = self.gym.get_actor_dof_properties(env_ptr, humanoid_handle)
            dof_prop['driveMode'].fill(gymapi.DOF_MODE_POS)
            dof_prop["stiffness"].fill(500.0)
            dof_prop["damping"].fill(50.0)
        
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

       
        dof_prop = self.gym.get_actor_dof_properties(env_ptr, humanoid_handle)
    
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        for i in range(len(extremity_names)):
            self.extremities_index[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], extremity_names[i])
        
        print(self.gym.get_actor_joint_names(self.envs[0], self.humanoid_handles[0]))

    def compute_reward(self, actions):
        # self.rew_buf[:], self.reset_buf[:] = compute_ant_reward(
        #     self.obs_buf,
        #     self.reset_buf,
        #     self.progress_buf,
        #     self.actions,
        #     self.up_weight,
        #     self.heading_weight,
        #     self.potentials,
        #     self.prev_potentials,
        #     self.actions_cost_scale,
        #     self.energy_cost_scale,
        #     self.joints_at_limit_cost_scale,
        #     self.termination_height,
        #     self.death_cost,
        #     self.max_episode_length
        # )
        pass

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        #print("Feet forces and torques: ", self.vec_sensor_tensor[0, :])
        # print(self.vec_sensor_tensor.shape)
        self.obs_buf[:] = compute_humanoid_observations(self.root_states,
            self.inv_start_rot, self.dof_pos, self.dof_vel, self.rb_pos,
            self.basis_vec0, self.basis_vec1)
        pass

    def reset(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        idx = env_ids.cpu().numpy()
        self.t[idx] = np.random.uniform(0, int(self.motion_generator.motion_time/self.dt), (idx.shape[0], )) * self.dt
        self.num_steps[idx] = 0

        pose, vel = self.motion_generator.generate_batch(self.t)

        self.dof_pos[env_ids] = torch.Tensor(pose[idx, 7:]).to(self.device)
        self.dof_vel[env_ids] = torch.Tensor(vel[idx, 6:]).to(self.device)
      
        self.root_states[env_ids, 0:7] = torch.Tensor(pose[idx, 0:7]).to(self.device)
        self.root_states[env_ids, 7:13] = torch.Tensor(vel[idx, 0:6]).to(self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

       
        self.reset_buf[env_ids] = 0

    def set_pose(self):
        #test
        pose, vel = self.motion_generator.generate_batch(self.num_steps * 1.0/60)
        self.dof_pos[:] = torch.Tensor(pose[:,7:]).to(self.device)
        self.dof_vel[:] = torch.Tensor(vel[:, 6:]).to(self.device)

        self.root_states[:,0:7] = torch.Tensor(pose[:, 0:7]).to(self.device)
        self.root_states[:,7:13] = torch.Tensor(vel[:,0:6]).to(self.device)
        #print(self.root_states)
        self.num_steps += 1
        #print(pose[0:7])
      
     
        self.gym.set_dof_state_tensor(self.sim,
                                        gymtorch.unwrap_tensor(self.dof_state))
        self.gym.set_actor_root_state_tensor(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states))
        self.gym.simulate(self.sim)
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
                                            
      

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        pose, _ = self.motion_generator.generate_batch(self.t)
        self.actions = self.actions + torch.Tensor(pose[:,7:]).to(self.device)
        actions = gymtorch.unwrap_tensor(self.actions)
        self.gym.set_dof_position_target_tensor(self.sim, actions)
        self.num_steps += 1
        self.t += 1.0/60
        self.t = self.t % self.motion_generator.motion_time 

    def post_physics_step(self):
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_actor_root_state_tensor(self.sim)

            points = []
            colors = []
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
                               glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy()])
                colors.append([0.97, 0.1, 0.06])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(), glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy()])
                colors.append([0.05, 0.99, 0.04])

            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)
        if((self.num_steps%100).any() == 0):
            self.reset_buf[:] = 1
            env_ids = torch.where(self.reset_buf>0)[0]
            self.reset(env_ids)

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_ant_reward(
    obs_buf,
    reset_buf,
    progress_buf,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    joints_at_limit_cost_scale,
    termination_height,
    death_cost,
    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # reward from direction headed
    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)

    # aligning up axis of ant and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 20:28]), dim=-1)
    dof_at_limit_cost = torch.sum(obs_buf[:, 12:20] > 0.99, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * 0.5
    progress_reward = potentials - prev_potentials

    total_reward = progress_reward + alive_reward + up_reward + heading_reward - \
        actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost * joints_at_limit_cost_scale

    # adjust reward for fallen agents
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)

    # reset agents
    #reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    #reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    reset = reset_buf

    return total_reward, reset


@torch.jit.script
def compute_humanoid_observations(root_states, inv_start_rot, dof_pos, dof_vel, rb_pos,
                             basis_vec0, basis_vec1):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    q = compute_coord(torso_rotation, inv_start_rot, basis_vec0, basis_vec1)

    pos_rfoot = rb_pos[:,4]
    pos_rfoot =  quat_rotate(q, pos_rfoot - torso_position)
    pos_lfoot = rb_pos[:,9]
    pos_lfoot =  quat_rotate(q, pos_lfoot - torso_position)
    pos_head = rb_pos[:,16]
    pos_head = pos_head - torso_position
    pos_lwrist = rb_pos[:,20]
    pos_lwrist =  quat_rotate(q, pos_lwrist - torso_position)
    pos_rwrist = rb_pos[:,24]
    pos_rwrist =  quat_rotate(q, pos_rwrist - torso_position)
    
    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs(8), num_dofs(8), 24, num_dofs(8)
    obs = torch.cat((torso_position[:,1:2], torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, pos_rfoot, pos_lfoot, pos_head, pos_lwrist, pos_rwrist), dim=-1)

    return obs