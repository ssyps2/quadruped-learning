from isaacgym.torch_utils import *
import torch
import numpy as np


# Command indices for Z1 arm (spherical coordinates)
INDEX_EE_POS_RADIUS_CMD = 0
INDEX_EE_POS_PITCH_CMD = 1
INDEX_EE_POS_YAW_CMD = 2
INDEX_EE_TIMING_CMD = 3
INDEX_EE_ROLL_CMD = 4
INDEX_EE_PITCH_CMD = 5
INDEX_EE_YAW_CMD = 6


class Z1ArmBaseFrameRewards:
    """
    Reward functions for the Z1 robotic arm.
    All rewards are computed in the arm base frame (world frame for standalone Z1).
    No leg-related rewards or B1 base frame transformations.
    """
    
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    ###########################
    ########## ARM ############
    ###########################

    def _reward_manip_pos_tracking(self):
        """
        Reward for end-effector position tracking in arm base frame.
        Tracks the commanded target position in spherical coordinates (radius, pitch, yaw).
        """
        # Get current EE position in world frame
        ee_idx = self.env.gym.find_actor_rigid_body_handle(
            self.env.envs[0], self.env.robot_actor_handles[0], "gripperStator"
        )
        ee_pos_world = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:, ee_idx, 0:3].view(self.env.num_envs, 3)
        
        # Get commanded target from spherical coordinates (radius, pitch, yaw)
        # commands[:, 0] = radius, commands[:, 1] = pitch, commands[:, 2] = yaw
        radius_cmd = self.env.commands[:, INDEX_EE_POS_RADIUS_CMD].view(-1, 1)
        pitch_cmd = self.env.commands[:, INDEX_EE_POS_PITCH_CMD].view(-1, 1)
        yaw_cmd = self.env.commands[:, INDEX_EE_POS_YAW_CMD].view(-1, 1)
        
        # Convert spherical to Cartesian target position
        x_target = radius_cmd * torch.cos(pitch_cmd) * torch.cos(yaw_cmd)
        y_target = radius_cmd * torch.cos(pitch_cmd) * torch.sin(yaw_cmd)
        z_target = -radius_cmd * torch.sin(pitch_cmd)  # negative because pitch down = positive z in arm convention
        ee_target = torch.cat([x_target, y_target, z_target], dim=1)
        
        # Compute position error and exponential reward
        ee_position_error = torch.sum(torch.square(ee_target - ee_pos_world), dim=1)
        ee_position_coeff = 15.0
        pos_rew = torch.exp(-ee_position_coeff * ee_position_error)
        return pos_rew

    def _reward_manip_ori_tracking(self):
        """
        Reward for end-effector orientation tracking in arm base frame.
        Uses the explicit orientation commands (roll, pitch, yaw) from indices 4, 5, 6.
        """
        # Get current EE orientation
        ee_idx = self.env.gym.find_actor_rigid_body_handle(
            self.env.envs[0], self.env.robot_actor_handles[0], "gripperStator"
        )
        ee_quat = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:, ee_idx, 3:7].view(self.env.num_envs, 4)
        ee_rpy = torch.stack(get_euler_xyz(ee_quat), dim=1)
        
        # Target orientation from explicit orientation commands
        target_roll = self.env.commands[:, INDEX_EE_ROLL_CMD]
        target_pitch = self.env.commands[:, INDEX_EE_PITCH_CMD]
        target_yaw = self.env.commands[:, INDEX_EE_YAW_CMD]
        
        roll_error = torch.minimum(
            torch.abs(ee_rpy[:, 0] - target_roll),
            2 * np.pi - torch.abs(ee_rpy[:, 0] - target_roll)
        )
        pitch_error = torch.minimum(
            torch.abs(ee_rpy[:, 1] - target_pitch),
            2 * np.pi - torch.abs(ee_rpy[:, 1] - target_pitch)
        )
        yaw_error = torch.minimum(
            torch.abs(ee_rpy[:, 2] - target_yaw),
            2 * np.pi - torch.abs(ee_rpy[:, 2] - target_yaw)
        )
        
        tracking_coef_manip_ori = 1.0
        ee_ori_tracking_error = roll_error + pitch_error + yaw_error
        return torch.exp(-ee_ori_tracking_error * tracking_coef_manip_ori)

    def _reward_torque_limits_arm(self):
        """
        Penalize torques too close to the limit for arm joints.
        For Z1, all DOFs (0:7) are arm DOFs.
        """
        soft_limit = self.env.cfg.rewards.soft_torque_limit_arm if hasattr(self.env.cfg.rewards, 'soft_torque_limit_arm') else 0.9
        return torch.sum(torch.square(
            (torch.abs(self.env.torques) - self.env.torque_limits * soft_limit).clip(min=0.)
        ), dim=1)

    def _reward_dof_vel_arm(self):
        """
        Penalize high joint velocities for arm.
        For Z1, all DOFs are arm DOFs.
        """
        # Exclude gripper (last DOF) from velocity penalty if desired
        return torch.sum(torch.square(self.env.dof_vel[:, :6]), dim=1)

    def _reward_dof_acc_arm(self):
        """
        Penalize joint accelerations for arm.
        """
        return torch.sum(torch.square(
            (self.env.last_dof_vel[:, :6] - self.env.dof_vel[:, :6]) / self.env.dt
        ), dim=1)

    def _reward_action_rate_arm(self):
        """
        Penalize changes in actions (action rate).
        """
        return torch.sum(torch.square(
            self.env.last_actions[:, :6] - self.env.actions[:, :6]
        ), dim=1)

    def _reward_action_smoothness_1_arm(self):
        """
        Penalize first-order action smoothness (jerk).
        """
        diff = torch.square(self.env.joint_pos_target[:, :6] - self.env.last_joint_pos_target[:, :6])
        diff = diff * (self.env.last_actions[:, :6] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2_arm(self):
        """
        Penalize second-order action smoothness (snap).
        """
        diff = torch.square(
            self.env.joint_pos_target[:, :6] 
            - 2 * self.env.last_joint_pos_target[:, :6] 
            + self.env.last_last_joint_pos_target[:, :6]
        )
        diff = diff * (self.env.last_actions[:, :6] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :6] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_dof_pos_limits_arm(self):
        """
        Penalize joint positions too close to limits.
        """
        out_of_limits = -(self.env.dof_pos[:, :6] - self.env.dof_pos_limits[:6, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos[:, :6] - self.env.dof_pos_limits[:6, 1]).clip(min=0.)  # upper limit
        return torch.sum(out_of_limits, dim=1)

    def _reward_torques(self):
        """
        Penalize total torque magnitude.
        """
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_dof_pos(self):
        """
        Penalize deviation from default joint positions.
        """
        return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def _reward_survival(self):
        """
        Reward for staying alive (not terminated).
        """
        return torch.ones(self.env.num_envs, device=self.env.device)

    def _reward_termination(self):
        """
        Penalize termination.
        """
        return self.env.reset_buf.float()

    def _reward_orientation(self):
        """
        Penalize base orientation deviation (for fixed-base Z1, this should be minimal).
        """
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        """
        Penalize base height deviation (for fixed-base Z1, not typically used).
        """
        base_height = self.env.root_states[self.env.robot_actor_idxs, 2]
        target_height = self.env.cfg.rewards.base_height_target if hasattr(self.env.cfg.rewards, 'base_height_target') else 0.7
        return torch.square(base_height - target_height)

    def _reward_ee_velocity(self):
        """
        Penalize high end-effector velocity for smooth motion.
        """
        ee_idx = self.env.gym.find_actor_rigid_body_handle(
            self.env.envs[0], self.env.robot_actor_handles[0], "gripperStator"
        )
        ee_vel = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:, ee_idx, 7:10].view(self.env.num_envs, 3)
        return torch.sum(torch.square(ee_vel), dim=1)

    def _reward_ee_acceleration(self):
        """
        Penalize high end-effector acceleration.
        """
        ee_idx = self.env.gym.find_actor_rigid_body_handle(
            self.env.envs[0], self.env.robot_actor_handles[0], "gripperStator"
        )
        ee_vel = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:, ee_idx, 7:10].view(self.env.num_envs, 3)
        
        if not hasattr(self, 'last_ee_vel'):
            self.last_ee_vel = ee_vel.clone()
        
        ee_acc = (ee_vel - self.last_ee_vel) / self.env.dt
        self.last_ee_vel = ee_vel.clone()
        
        return torch.sum(torch.square(ee_acc), dim=1)

    def _reward_collision(self):
        """
        Penalize self-collisions or collisions with environment.
        """
        if hasattr(self.env, 'penalised_contact_indices') and len(self.env.penalised_contact_indices) > 0:
            return torch.sum(
                1.0 * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                dim=1
            )
        return torch.zeros(self.env.num_envs, device=self.env.device)

    def _reward_joint_regularization(self):
        """
        Regularization reward to keep joints near default/comfortable positions.
        """
        # Encourage arm to stay near default configuration
        joint_deviation = torch.sum(torch.square(self.env.dof_pos[:, :6] - self.env.default_dof_pos[:, :6]), dim=1)
        return torch.exp(-0.5 * joint_deviation)

    def _reward_manipulability(self):
        """
        Reward for maintaining good manipulability (avoiding singularities).
        Simplified version based on joint positions away from limits.
        """
        # Calculate how centered joints are within their limits
        joint_range = self.env.dof_pos_limits[:6, 1] - self.env.dof_pos_limits[:6, 0]
        joint_center = (self.env.dof_pos_limits[:6, 1] + self.env.dof_pos_limits[:6, 0]) / 2
        normalized_pos = (self.env.dof_pos[:, :6] - joint_center) / (joint_range / 2 + 1e-6)
        
        # Reward being away from limits (near center)
        manipulability = 1.0 - torch.mean(torch.square(normalized_pos), dim=1)
        return manipulability.clip(min=0.)
