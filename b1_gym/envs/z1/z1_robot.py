from b1_gym.envs.base.legged_robot import LeggedRobot
from b1_gym.rewards.z1_arm_base_rewards import Z1ArmBaseFrameRewards

class Z1Robot(LeggedRobot):
    """
    Z1Robot environment class for the z1 robot, using z1-specific reward logic and configuration.
    Inherits all simulation logic from LeggedRobot, but can override methods if needed for z1-specific behavior.
    """
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless,
                 initial_dynamics_dict=None, terrain_props=None, custom_heightmap=None):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,
                         initial_dynamics_dict, terrain_props, custom_heightmap)
        # Use z1-specific reward container
        self.reward_container = Z1ArmBaseFrameRewards(self)

    # If z1 needs to override any methods, do so here.
    # Otherwise, it will use the base LeggedRobot logic.
