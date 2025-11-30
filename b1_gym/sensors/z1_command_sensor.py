from .sensor import Sensor
import torch


class Z1CommandSensor(Sensor):
    """
    Sensor that provides the Z1 arm target commands to the policy.
    Returns the spherical position commands (radius, pitch, yaw) and optionally orientation.
    
    Command indices:
        0: radius
        1: pos_pitch  
        2: pos_yaw
        3: timing (unused)
        4: ori_roll
        5: ori_pitch
        6: ori_yaw
    """
    
    def __init__(self, env, attached_robot_asset=None, include_orientation=False):
        super().__init__(env)
        self.env = env
        self.include_orientation = include_orientation
        
    def get_observation(self, env_ids=None):
        # Return position commands (radius, pitch, yaw) - indices 0, 1, 2
        pos_commands = self.env.commands[:, 0:3]
        
        if self.include_orientation:
            # Also return orientation commands (roll, pitch, yaw) - indices 4, 5, 6
            ori_commands = self.env.commands[:, 4:7]
            return torch.cat([pos_commands, ori_commands], dim=1)
        
        return pos_commands
    
    def get_noise_vec(self):
        if self.include_orientation:
            return torch.zeros(6, device=self.env.device)
        return torch.zeros(3, device=self.env.device)
    
    def get_dim(self):
        return 6 if self.include_orientation else 3
