import torch
import numpy as np


class Z1Logger:
    """
    Logger for Z1 arm-only robot training.
    Adapted from Logger but without leg/feet-specific logging.
    """
    def __init__(self, env):
        self.env = env

    def populate_log(self, env_ids):
        extras = {}

        # fill extras
        if len(env_ids) > 0:
            extras["train/episode"] = {}
            for key in self.env.episode_sums.keys():
                extras["train/episode"]['rew_' + key] = torch.mean(
                    self.env.episode_sums[key][env_ids])
                self.env.episode_sums[key][env_ids] = 0.

        # log additional curriculum info
        if self.env.cfg.terrain.curriculum:
            extras["train/episode"]["terrain_level"] = torch.mean(
                self.env.terrain_levels.float())
        
        if self.env.cfg.commands.command_curriculum:
            commands = self.env.commands
            extras["env_bins"] = torch.Tensor(self.env.env_command_bins)
            
            extras["curriculum/distribution"] = {}
            for curriculum, category in zip(self.env.curricula, self.env.category_names):
                extras[f"curriculum/distribution"][f"weights_{category}"] = curriculum.weights
                extras[f"curriculum/distribution"][f"grid_{category}"] = curriculum.grid

        extras["time_outs"] = self.env.time_out_buf
        extras["train/episode"]["Number of env. terminated on time_outs"] = len(self.env.time_out_buf.nonzero(as_tuple=False).flatten())
        extras["train/episode"]["Number of env. terminated on contacts"] = len(self.env.contact_buf.nonzero(as_tuple=False).flatten())
        
        if self.env.cfg.rewards.use_terminal_body_height:
            extras["train/episode"]["Number of env. terminated on body_height"] = len(self.env.body_height_buf.nonzero(as_tuple=False).flatten())
        if self.env.cfg.rewards.use_terminal_roll_pitch:
            extras["train/episode"]["Number of env. terminated on body_roll_pitch"] = len(self.env.body_ori_buf.nonzero(as_tuple=False).flatten())
        if self.env.cfg.rewards.use_terminal_torque_arm_limits:
            extras["train/episode"]["Number of env. terminated on torque_arm_limits"] = len(self.env.arm_torque_lim_buff.nonzero(as_tuple=False).flatten())

        # Z1-specific logging (arm torques, no leg torques)
        extras["train/episode"]["arm torques"] = torch.sum(torch.square(self.env.torques), dim=1).mean()
        extras["train/episode"]["mean arm torques"] = torch.mean(torch.abs(self.env.torques))
        extras["train/episode"]["max arm torques"] = torch.max(torch.abs(self.env.torques))

        # Joint position tracking
        if hasattr(self.env, 'dof_pos'):
            extras["train/episode"]["mean joint pos"] = torch.mean(torch.abs(self.env.dof_pos))
            extras["train/episode"]["max joint pos"] = torch.max(torch.abs(self.env.dof_pos))
        
        # Joint velocity tracking
        if hasattr(self.env, 'dof_vel'):
            extras["train/episode"]["mean joint vel"] = torch.mean(torch.abs(self.env.dof_vel))
            extras["train/episode"]["max joint vel"] = torch.max(torch.abs(self.env.dof_vel))

        # End-effector tracking (if available)
        if hasattr(self.env, 'ee_pos_target') and hasattr(self.env, 'rigid_body_state'):
            # Compute EE position error
            ee_pos = self.env.rigid_body_state[:, self.env.gripper_stator_index, :3]
            ee_error = torch.norm(ee_pos - self.env.ee_pos_target, dim=-1)
            extras["train/episode"]["EE position error"] = torch.mean(ee_error)
            extras["train/episode"]["EE position error max"] = torch.max(ee_error)

        # Command logging for Z1 (simplified - EE position commands)
        if hasattr(self.env, 'commands') and self.env.commands.shape[1] >= 3:
            commands = self.env.commands
            extras["train/episode"]["command_ee_x"] = torch.mean(commands[:, 0])
            extras["train/episode"]["command_ee_y"] = torch.mean(commands[:, 1])
            extras["train/episode"]["command_ee_z"] = torch.mean(commands[:, 2])

        extras["train/episode"]['Number of env. reset'] = len(env_ids)
        extras["train/episode"]['Average episode length'] = torch.mean(self.env.episode_length_buf[env_ids] * self.env.dt)

        return extras
