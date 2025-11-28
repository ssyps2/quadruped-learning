"""
Z1Robot: Environment class for the Unitree Z1 robotic arm.
This is a simplified version of LeggedRobot adapted for arm-only control.
"""

from b1_gym.envs.base.legged_robot import LeggedRobot
from b1_gym.rewards.z1_arm_base_rewards import Z1ArmBaseFrameRewards
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import torch
import numpy as np


class Z1Robot(LeggedRobot):
    """
    Z1Robot environment class for the z1 robot, using z1-specific reward logic and configuration.
    Inherits simulation logic from LeggedRobot but overrides methods for arm-only behavior.
    
    Key differences from LeggedRobot:
    - No leg DOFs (only 7 arm DOFs)
    - No foot contacts or gait patterns
    - Simplified command structure
    - Arm-specific reward functions
    """
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless,
                 initial_dynamics_dict=None, terrain_props=None, custom_heightmap=None):
        # Call parent constructor
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,
                         initial_dynamics_dict, terrain_props, custom_heightmap)
        # Use z1-specific reward container
        self.reward_container = Z1ArmBaseFrameRewards(self)

    def step(self, actions):
        """
        Override step to return 4 values as expected by HistoryWrapper.
        Puts privileged_obs into extras dict instead of returning separately.
        """
        # Call parent step which returns 5 values
        obs, privileged_obs, rew, done, extras = LeggedRobot.step(self, actions)
        
        # Put privileged_obs into extras for HistoryWrapper compatibility
        extras["privileged_obs"] = privileged_obs
        
        # Add env_bins if not present (required by PPO)
        if "env_bins" not in extras:
            extras["env_bins"] = torch.zeros(self.num_envs, device=self.device)
        
        return obs, rew, done, extras

    def reset(self):
        """
        Override reset to use the parent's step directly (5 values) then return properly.
        """
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # Use parent's step directly which returns 5 values
        obs, privileged_obs, _, _, _ = LeggedRobot.step(
            self, torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def _get_body_indices(self, body_names):
        """
        Override to handle Z1-specific body naming.
        Z1 URDF has: base (world), link00-link06, gripperStator, gripperMover
        """
        # Find gripperStator index (for end-effector)
        gripper_indices = [i for i, name in enumerate(body_names) if name == "gripperStator"]
        self.gripper_stator_index = gripper_indices[0] if gripper_indices else -1
        
        # Find base index - for Z1, link00 is the arm base
        base_indices = [i for i, name in enumerate(body_names) if name == "link00"]
        if not base_indices:
            # Try "base" as fallback
            base_indices = [i for i, name in enumerate(body_names) if name == "base"]
        self.robot_base_index = base_indices[0] if base_indices else 0

    def _process_dof_props(self, props, env_id):
        """
        Override for Z1 arm-only robot. All DOFs are arm DOFs (no legs).
        Sets up position limits, velocity limits, torque limits, and PD gains.
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_damping = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_friction = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                self.dof_damping[i] = props["damping"][i].item()
                self.dof_friction[i] = props["friction"][i].item()

                # Z1: all DOFs are arm DOFs, use arm gains
                if self.cfg.asset.default_dof_drive_mode == 0:  # mixed (effort for arm)
                    props["stiffness"][i] = 0
                    props["damping"][i] = 0
                    props['driveMode'] = gymapi.DOF_MODE_EFFORT
                elif self.cfg.asset.default_dof_drive_mode == 1:  # position control
                    props["stiffness"][i] = self.cfg.commands.p_gains_arm[i]
                    props["damping"][i] = self.cfg.commands.d_gains_arm[i]
                    props['driveMode'] = gymapi.DOF_MODE_POS
                elif self.cfg.asset.default_dof_drive_mode == 3:  # effort
                    props["stiffness"][i] = 0
                    props["damping"][i] = 0
                    props['driveMode'] = gymapi.DOF_MODE_EFFORT
                else:
                    raise Exception(f"Drive mode {self.cfg.asset.default_dof_drive_mode} not found!")

                # Soft limits for arm DOFs
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        return props

    def _resample_force_or_position_control(self, env_ids):
        """
        Override for Z1: no force/position hybrid control needed for arm-only robot.
        Just set force_or_position_control to position mode (0).
        """
        self.force_or_position_control[env_ids] = 0

    def _init_buffers(self):
        """
        Override to handle arm-only buffers. Calls parent but handles arm-specific initialization.
        """
        # Get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # Create wrapper tensors
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.base_pos = self.root_states[self.robot_actor_idxs, 0:3]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[self.robot_actor_idxs, 3:7]
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs*self.total_rigid_body_num,:].view(self.num_envs,self.total_rigid_body_num,13)[:,0:self.num_bodies, :]
        self.rigid_body_state_object = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs*self.total_rigid_body_num,:].view(self.num_envs,self.total_rigid_body_num,13)[:,self.num_bodies:self.num_bodies + self.num_object_bodies, :]
        
        # For Z1, we may not have feet - create empty tensors if feet_indices is empty
        if len(self.feet_indices) > 0:
            self.foot_velocities = self.rigid_body_state.view(self.num_envs, -1, 13)[:, self.feet_indices, 7:10]
            self.foot_positions = self.rigid_body_state.view(self.num_envs, -1, 13)[:, self.feet_indices, 0:3]
        else:
            self.foot_velocities = torch.zeros(self.num_envs, 1, 3, dtype=torch.float, device=self.device)
            self.foot_positions = torch.zeros(self.num_envs, 1, 3, dtype=torch.float, device=self.device)
        
        # Gripper position (end effector)
        if hasattr(self, 'gripper_stator_index') and self.gripper_stator_index >= 0:
            self.gripper_position = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_stator_index, 0:3]
            self.gripper_velocity = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_stator_index, 7:10]
        else:
            self.gripper_position = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
            self.gripper_velocity = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
            
        self.prev_base_pos = self.base_pos.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs*self.total_rigid_body_num,:].view(self.num_envs,self.total_rigid_body_num,3)[:,0:self.num_bodies, :]

        # Object handling (if balls/objects are present)
        if self.cfg.env.add_balls:
            self.object_pos_world_frame = self.root_states[self.object_actor_idxs, 0:3]
            robot_object_vec = self.asset.get_local_pos()
            self.object_local_pos = quat_rotate_inverse(self.base_quat, robot_object_vec)
            self.object_local_pos[:, 2] = 0.0
            self.last_object_local_pos = torch.clone(self.object_local_pos)
            self.object_lin_vel = self.asset.get_lin_vel()
            self.object_ang_vel = self.asset.get_ang_vel()

        # Initialize common data
        self.common_step_counter = 0
        self.extras = {}

        self.height_points = self._init_height_points(torch.arange(self.num_envs, device=self.device), self.cfg)
        self.measured_heights = 0

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.forces = torch.zeros(self.num_envs, self.total_rigid_body_num, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Reward tracking
        self.save_rew_feet_contact = torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=False)
        self.save_rew_torque = torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=False)
        self.save_ee_force_magnitude = torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=False)
        self.save_ee_force_z = torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=False)
        self.save_ee_force_direction_angle = torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=False)

        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[self.robot_actor_idxs, 7:13])
        self.path_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.past_base_pos = self.base_pos.clone()

        # Commands buffer
        self.commands_value = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                          device=self.device, requires_grad=False)
        self.commands = torch.zeros_like(self.commands_value)
        self.heading_commands = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.heading_offsets = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # Push buffers (may not be used for arm but needed for compatibility)
        self.selected_env_ids = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        push_min = int(self.cfg.domain_rand.push_interval_gripper_min) if hasattr(self.cfg.domain_rand, 'push_interval_gripper_min') else 1
        push_max = int(self.cfg.domain_rand.push_interval_gripper_max) if hasattr(self.cfg.domain_rand, 'push_interval_gripper_max') else 10
        if push_max <= push_min:
            push_max = push_min + 1
        self.push_interval = torch.randint(push_min, push_max, (self.num_envs, 1), device=self.device, requires_grad=False)
        self.push_end_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_target = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_forces_eval_time = torch.zeros(1, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.current_Fxyz_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_force = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_force_kps = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_force_kds = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.selected_env_ids_robot = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.push_end_time_robot = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_force_robot = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        # Force/position control (Z1 uses position only)
        self.force_or_position_control = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._resample_force_or_position_control(torch.arange(self.num_envs, device=self.device))

        # Gripper tracking buffers
        self.gripper_pos_tracking_error_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_ori_tracking_error_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # Command scales (simplified for Z1)
        self.commands_scale = torch.tensor([
            self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel
        ], device=self.device, requires_grad=False)[:self.cfg.commands.num_commands]

        # Contact states (may be empty for arm-only)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        num_feet = len(self.feet_indices) if len(self.feet_indices) > 0 else 1
        self.feet_air_time = torch.zeros(self.num_envs, num_feet, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, num_feet, dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contact_filt = torch.zeros(self.num_envs, num_feet, dtype=torch.bool, device=self.device, requires_grad=False)
        
        # Clock inputs for gait (not used for arm, but needed for ClockSensor compatibility)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Gait indices (not used for arm, but needed for compatibility)
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Default DOF positions and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if name in self.cfg.init_state.default_joint_angles:
                angle = self.cfg.init_state.default_joint_angles[name]
                self.default_dof_pos[i] = angle
            
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.

        # Override with arm-specific gains if provided
        if hasattr(self.cfg.commands, 'p_gains_arm') and len(self.cfg.commands.p_gains_arm) > 0:
            num_arm_dofs = min(len(self.cfg.commands.p_gains_arm), self.num_dof)
            self.p_gains[:num_arm_dofs] = torch.tensor(self.cfg.commands.p_gains_arm[:num_arm_dofs], device=self.device)
        if hasattr(self.cfg.commands, 'd_gains_arm') and len(self.cfg.commands.d_gains_arm) > 0:
            num_arm_dofs = min(len(self.cfg.commands.d_gains_arm), self.num_dof)
            self.d_gains[:num_arm_dofs] = torch.tensor(self.cfg.commands.d_gains_arm[:num_arm_dofs], device=self.device)

        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
        # Kp/Kd factors for domain randomization
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        # Initialize custom buffers
        self._init_custom_buffers__()
        
        # Override logger with Z1-specific logger
        from b1_gym.utils.z1_logger import Z1Logger
        self.logger = Z1Logger(self)

    def check_termination(self):
        """
        Override to remove leg-specific termination conditions.
        For Z1 arm, we mainly check for timeouts and arm-specific limits.
        """
        self.contact_buf = torch.zeros_like(self.reset_buf, dtype=torch.bool)
        self.reset_buf = torch.clone(self.contact_buf)
        
        # Timeout
        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length
        self.reset_buf |= self.time_out_buf

        # Arm torque limits (if configured)
        if hasattr(self.cfg.rewards, 'use_terminal_torque_arm_limits') and self.cfg.rewards.use_terminal_torque_arm_limits:
            soft_limit = self.cfg.rewards.soft_torque_limit_arm if hasattr(self.cfg.rewards, 'soft_torque_limit_arm') else 0.9
            above_torque_lim = torch.any(
                (torch.abs(self.torques) - self.torque_limits * soft_limit) > 0.0, 
                dim=1
            ).view(self.num_envs)
            min_time = self.cfg.rewards.termination_torque_min_time if hasattr(self.cfg.rewards, 'termination_torque_min_time') else 0
            sim_started_while_ago = self.episode_length_buf > min_time
            self.arm_torque_lim_buff = torch.logical_and(above_torque_lim, sim_started_while_ago)
            self.reset_buf = torch.logical_or(self.arm_torque_lim_buff, self.reset_buf)

        # EE position limits (if configured)
        if hasattr(self.cfg.rewards, 'use_terminal_ee_position') and self.cfg.rewards.use_terminal_ee_position:
            terminal_dist = self.cfg.rewards.terminal_ee_distance if hasattr(self.cfg.rewards, 'terminal_ee_distance') else 1.0
            ee_position_outside_box = (
                torch.norm(self.gripper_position - self.ee_init_pos_world, dim=1) > terminal_dist
            ).view(self.num_envs)
            min_time = self.cfg.rewards.termination_torque_min_time if hasattr(self.cfg.rewards, 'termination_torque_min_time') else 0
            sim_started_while_ago = self.episode_length_buf > min_time
            self.ee_position_lim_buff = torch.logical_and(ee_position_outside_box, sim_started_while_ago)
            self.reset_buf = torch.logical_or(self.ee_position_lim_buff, self.reset_buf)

    def reset_idx(self, env_ids):
        """
        Override reset_idx for Z1 arm-only robot.
        Removes force control handling and leg-specific resets.
        """
        if len(env_ids) == 0:
            return

        # Update curriculum (if enabled)
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # Reset robot states
        self._resample_commands(env_ids)
        self._randomize_dof_props(env_ids, self.cfg)
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._randomize_rigid_body_props(env_ids, self.cfg)
            self.refresh_actor_rigid_shape_props(env_ids, self.cfg)

        self._reset_dofs(env_ids, self.cfg)
        self._reset_root_states(env_ids, self.cfg)

        # Reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.path_distance[env_ids] = 0.
        self.past_base_pos[env_ids] = self.base_pos.clone()[env_ids]
        self.reset_buf[env_ids] = 1
        self.torques[env_ids] = 0

        # Reset history buffers (if present)
        if hasattr(self, "obs_history"):
            self.obs_history_buf[env_ids, :] = 0
            self.obs_history[env_ids, :] = 0

        self.extras = self.logger.populate_log(env_ids)
        self.episode_length_buf[env_ids] = 0

        if hasattr(self, 'gait_indices'):
            self.gait_indices[env_ids] = 0

        # Reset lag buffer
        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids] = 0

    def _compute_torques(self, actions):
        """
        Compute torques from actions for arm-only robot.
        Simplified version without leg-specific scaling.
        """
        actions_scaled = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        actions_scaled[:, :self.num_actions] = actions[:, :self.num_actions] * self.cfg.control.action_scale

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            actions_post = self.lag_buffer[0]
        else:
            actions_post = actions_scaled

        if self.cfg.control.control_type == "P":
            self.joint_pos_target = actions_post + self.default_dof_pos
        elif self.cfg.control.control_type == "dP":
            self.joint_pos_target += actions_post * self.dt / self.cfg.control.decimation

        # Compute torques with actuator model
        dof_pos_error = self.dof_pos - self.joint_pos_target
        dof_vel = self.dof_vel

        ideal_torques = -self.p_gains * self.Kp_factors * dof_pos_error - self.d_gains * self.Kd_factors * dof_vel
        stall_torque = self.torque_limits
        max_vel = self.dof_vel_limits
        
        # Avoid division by zero
        max_vel_safe = torch.clamp(max_vel, min=1e-6)
        vel_torque_max = stall_torque * (1 - torch.clip(torch.abs(dof_vel / max_vel_safe), max=1))
        vel_torque_min = -stall_torque * (1 - torch.clip(torch.abs(dof_vel / max_vel_safe), max=1))

        clipped_torques = ideal_torques.clone()
        clipped_torques = clipped_torques.clamp(vel_torque_min, vel_torque_max)
        self.torques = clipped_torques

    def pre_physics_step(self):
        """
        Override to handle arm-only case (no feet).
        """
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        if len(self.feet_indices) > 0:
            self.prev_foot_velocities = self.foot_velocities.clone()
        self.render_gui()

    def _resample_commands(self, env_ids):
        """
        Simplified command resampling for Z1 arm.
        Override if Z1 needs specific command handling.
        """
        if len(env_ids) == 0:
            return
        
        # Simple velocity commands for now
        self.commands[env_ids, 0] = torch_rand_float(
            self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_x[1], 
            (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.cfg.commands.lin_vel_y[0], self.cfg.commands.lin_vel_y[1], 
            (len(env_ids), 1), device=self.device
        ).squeeze(1)
        if self.cfg.commands.num_commands > 2:
            self.commands[env_ids, 2] = torch_rand_float(
                self.cfg.commands.ang_vel_yaw[0], self.cfg.commands.ang_vel_yaw[1], 
                (len(env_ids), 1), device=self.device
            ).squeeze(1)

    def _post_physics_step_callback(self):
        """
        Simplified post physics step for arm-only robot.
        """
        # Resample commands if needed
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

    def _reset_dofs(self, env_ids, cfg):
        """
        Override for Z1: Reset DOF positions and velocities for arm-only robot.
        No leg DOFs to handle - only 7 arm DOFs.
        """
        # Reset to default positions with small random noise
        self.dof_pos[env_ids] = self.default_dof_pos
        # Add small random noise to arm joint positions (exclude gripper at index 6)
        self.dof_pos[env_ids, :6] += torch_rand_float(-0.1, 0.1, (len(env_ids), 6), device=self.device)
        self.joint_pos_target[env_ids] = self.dof_pos[env_ids]
        
        # Reset velocities to zero
        self.dof_vel[env_ids] = 0.

        # Update simulation state
        all_subject_env_ids = self.robot_actor_idxs[env_ids].to(device=self.device)
        all_subject_env_ids_int32 = all_subject_env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(all_subject_env_ids_int32), 
            len(all_subject_env_ids_int32)
        )

    def _step_contact_targets(self):
        """
        No contact targets for arm-only robot.
        """
        pass

    def _push_robots(self, env_ids, cfg):
        """
        Override to skip robot pushing for arm-only setup.
        """
        pass

    def _push_gripper(self, env_ids_all, cfg):
        """
        Override to skip gripper pushing for basic arm control.
        """
        pass

    def _push_robot_base(self, env_ids, cfg):
        """
        Override to skip base pushing for arm-only setup.
        """
        pass

    def _create_envs(self):
        """
        Override to handle Z1-specific body naming.
        Z1 URDF has: base (world), link00-link06, gripperStator, gripperMover
        """
        # Call parent to create envs
        super()._create_envs()
        
        # Fix body indices for Z1
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        
        # Find gripperStator index (for end-effector)
        gripper_indices = [i for i, name in enumerate(body_names) if name == "gripperStator"]
        if gripper_indices:
            self.gripper_stator_index = gripper_indices[0]
        else:
            # Fallback to link06 if gripperStator not found
            gripper_indices = [i for i, name in enumerate(body_names) if name == "link06"]
            self.gripper_stator_index = gripper_indices[0] if gripper_indices else 0
        
        # Find base index - for Z1, link00 is the arm base
        base_indices = [i for i, name in enumerate(body_names) if name == "link00"]
        if base_indices:
            self.robot_base_index = base_indices[0]
        else:
            # Try "base" as fallback
            base_indices = [i for i, name in enumerate(body_names) if name == "base"]
            self.robot_base_index = base_indices[0] if base_indices else 0

    def get_measured_ee_pos_spherical(self) -> torch.Tensor:
        """
        Override for Z1: Get end-effector position in spherical coordinates.
        For standalone Z1, commands are at indices 0, 1, 2 (radius, pitch, yaw).
        The arm base is at the world origin (no B1 body to offset from).
        """
        # Get EE position in world frame
        ee_position_world = self.rigid_body_state[:, self.gripper_stator_index, :3].view(self.num_envs, 3)
        
        # For standalone Z1, the arm base is at the origin
        # So EE position in arm frame = EE position in world frame (approximately)
        ee_position_arm = ee_position_world.clone()
        
        # Convert to spherical coordinates
        radius = torch.norm(ee_position_arm, dim=1).view(self.num_envs, 1)
        # Avoid division by zero
        radius_safe = torch.clamp(radius, min=1e-6)
        pitch = -torch.asin(torch.clamp(ee_position_arm[:, 2:3] / radius_safe, -1.0, 1.0))
        yaw = torch.atan2(ee_position_arm[:, 1:2], ee_position_arm[:, 0:1])
        
        ee_pos_sphe = torch.cat((radius, pitch, yaw), dim=1).view(self.num_envs, 3)
        
        # Compute tracking error using Z1 command indices (0, 1, 2)
        radius_cmd = self.commands[:, 0].view(self.num_envs, 1)
        pitch_cmd = self.commands[:, 1].view(self.num_envs, 1)
        yaw_cmd = self.commands[:, 2].view(self.num_envs, 1)
        
        # Convert commanded spherical to cartesian
        x_cmd = radius_cmd * torch.cos(pitch_cmd) * torch.cos(yaw_cmd)
        y_cmd = radius_cmd * torch.cos(pitch_cmd) * torch.sin(yaw_cmd)
        z_cmd = -radius_cmd * torch.sin(pitch_cmd)
        ee_position_cmd = torch.cat((x_cmd, y_cmd, z_cmd), dim=1)
        
        # Compute position tracking error
        self.gripper_pos_tracking_error_buf = torch.norm(ee_position_cmd - ee_position_arm, dim=1)
        
        return ee_pos_sphe

    def compute_energy(self):
        """
        Override for Z1: Compute energy consumption for arm joints only.
        Z1 has 7 DOFs (6 arm joints + 1 gripper).
        """
        torques = self.torques  # All 7 DOFs
        joint_vels = self.dof_vel
        
        # Gear ratios for Z1 arm (approximate - all joints similar)
        gear_ratios = torch.ones(self.num_dof, device=self.device)
        
        power_joule = torch.sum((torques * gear_ratios)**2 * 0.07, dim=1)
        power_mechanical = torch.sum(torch.clip(torques * joint_vels, -3, 10000), dim=1)
        power_battery = 24.0  # Lower battery for arm-only
        
        return power_joule + power_mechanical + power_battery
