from isaacgym.torch_utils import *
import torch
import numpy as np


INDEX_EE_POS_RADIUS_CMD = 15
INDEX_EE_POS_PITCH_CMD = 16
INDEX_EE_POS_YAW_CMD = 17
INDEX_EE_ROLL_CMD = 19
INDEX_EE_PITCH_CMD = 20
INDEX_EE_YAW_CMD = 21

class Z1ArmBaseFrameRewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def _reward_manip_pos_tracking(self):
        '''
        Reward for manipulation tracking (EE position) in arm base frame
        '''
        # Commands in spherical coordinates in the arm base frame 
        radius_cmd = self.env.commands[:, INDEX_EE_POS_RADIUS_CMD].view(self.env.num_envs, 1) 
        pitch_cmd = self.env.commands[:, INDEX_EE_POS_PITCH_CMD].view(self.env.num_envs, 1) 
        yaw_cmd = self.env.commands[:, INDEX_EE_POS_YAW_CMD].view(self.env.num_envs, 1) 

        # Spherical to cartesian coordinates in the arm base frame 
        x_cmd_arm = radius_cmd * torch.cos(pitch_cmd) * torch.cos(yaw_cmd)
        y_cmd_arm = radius_cmd * torch.cos(pitch_cmd) * torch.sin(yaw_cmd)
        z_cmd_arm = -radius_cmd * torch.sin(pitch_cmd)
        ee_position_cmd_arm = torch.cat((x_cmd_arm, y_cmd_arm, z_cmd_arm), dim=1)

        # Get current ee position in world frame 
        ee_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperStator")
        ee_pos_world = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:, ee_idx, 0:3].view(self.env.num_envs, 3)

        # Since z1 base coincides with world frame, use world frame directly
        ee_position_error = torch.sum(torch.square(ee_position_cmd_arm - ee_pos_world), dim=1)
        ee_position_coeff = 15.0
        pos_rew = torch.exp(-ee_position_coeff * ee_position_error)
        return pos_rew

    def _reward_manip_ori_tracking(self):
        '''
        Reward for manipulation orientation tracking in arm base frame
        '''
        ee_rpy_yrf = self.env.get_measured_ee_rpy_yrf()  # Should be in arm base frame
        ee_ori_cmd = self.env.commands[:, INDEX_EE_ROLL_CMD:INDEX_EE_YAW_CMD+1].clone()
        roll_error = torch.minimum(torch.abs(ee_rpy_yrf[:, 0] - ee_ori_cmd[:, 0]), 2 * np.pi - torch.abs(ee_rpy_yrf[:, 0] - ee_ori_cmd[:, 0]))
        pitch_error = torch.minimum(torch.abs(ee_rpy_yrf[:, 1] - ee_ori_cmd[:, 1]), 2 * np.pi - torch.abs(ee_rpy_yrf[:, 1] - ee_ori_cmd[:, 1]))
        yaw_error = torch.minimum(torch.abs(ee_rpy_yrf[:, 2] - ee_ori_cmd[:, 2]), 2 * np.pi - torch.abs(ee_rpy_yrf[:, 2] - ee_ori_cmd[:, 2]))
        tracking_coef_manip_ori = 1.0  # Set as needed
        ee_ori_tracking_error = roll_error + pitch_error + yaw_error
        return torch.exp(-ee_ori_tracking_error * tracking_coef_manip_ori)

    # Add more rewards as needed, always using arm base frame for all calculations.
