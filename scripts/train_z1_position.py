import isaacgym
assert isaacgym
import numpy as np
from params_proto import PrefixProto
import torch

from b1_gym.envs.z1.z1_config import config_z1
from b1_gym.envs.z1.z1_robot import Z1Robot
from b1_gym_learn.ppo_cse import Runner
from b1_gym_learn.ppo_cse.actor_critic import AC_Args
from b1_gym_learn.ppo_cse.ppo import PPO_Args
from b1_gym_learn.ppo_cse import RunnerArgs

class RunCfg(PrefixProto, cli=False):
    experiment_group = "z1_position_control"
    experiment_job_type = "default"

def configure_env():
    from b1_gym.envs.base.legged_robot_config import Cfg
    config_z1(Cfg)

    # Z1-specific overrides and best practices
    Cfg.env.num_envs = 30
    Cfg.env.num_scalar_observations = 42
    Cfg.env.num_observations = 42
    Cfg.env.episode_length_s = 20
    Cfg.sim.physx.max_gpu_contact_pairs = 2 ** 25
    Cfg.robot.name = "z1"

    # Sensors (adjust as needed for z1)
    Cfg.sensors.sensor_names = [
      "OrientationSensor",
      "RCSensor",
      "JointPositionSensor",
      "JointVelocitySensor",
      "ActionSensor",
      "ClockSensor",
    ]
    Cfg.sensors.sensor_args = {
      "OrientationSensor": {},
      "RCSensor": {},
      "JointPositionSensor": {},
      "JointVelocitySensor": {},
      "ActionSensor": {},
      "ClockSensor": {},
    }

    # Privileged observations (if any)
    Cfg.sensors.privileged_sensor_names = []
    Cfg.sensors.privileged_sensor_args = {}
    Cfg.env.num_privileged_obs = 0

    # Reward container
    Cfg.rewards.reward_container_name = "Z1ArmBaseFrameRewards"

    # Domain randomization (adjust as needed)
    Cfg.domain_rand.rand_interval_s = 6
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [-1, 3]
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.max_push_vel_xy = 0.5
    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [0.05, 4.5]
    Cfg.domain_rand.randomize_restitution = True
    Cfg.domain_rand.restitution_range = [0.0, 1.0]
    Cfg.domain_rand.restitution = 0.5
    Cfg.domain_rand.randomize_com_displacement = True
    Cfg.domain_rand.com_displacement_range = [-0.1, 0.1]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.Kp_factor_range = [0.8, 1.3]
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.Kd_factor_range = [0.5, 1.5]

    # Rewards (adjust as needed)
    Cfg.reward_scales.manip_pos_tracking = 3.0
    Cfg.reward_scales.manip_ori_tracking = 2.5
    Cfg.reward_scales.torque_limits_arm = -0.005
    Cfg.reward_scales.dof_vel_arm = -0.003
    Cfg.reward_scales.dof_acc_arm = -3e-8
    Cfg.reward_scales.action_rate_arm = 0.0
    Cfg.reward_scales.action_smoothness_1_arm = -0.01
    Cfg.reward_scales.action_smoothness_2_arm = -0.01
    Cfg.reward_scales.dof_pos_limits_arm = -10.0
    Cfg.rewards.only_positive_rewards = True
    Cfg.rewards.total_rew_scale = 0.2

    # PD gains (example, adjust as needed)
    Cfg.commands.p_gains_arm = [64., 128., 64., 64., 64., 64., 64.]
    Cfg.commands.d_gains_arm = [1.5, 3.0, 1.5, 1.5, 1.5, 1.5, 1.5]
    Cfg.control.decimation = 4

    # Terrain
    Cfg.terrain.mesh_type = 'trimesh'
    Cfg.terrain.terrain_noise_magnitude = 0.0
    Cfg.terrain.teleport_robots = True
    Cfg.terrain.border_size = 50
    Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    Cfg.terrain.curriculum = False

    # Viewer
    Cfg.viewer.follow_robot = False
    Cfg.env.recording_width_px = 180
    Cfg.env.recording_height_px = 120

    return Cfg

def train_z1_position_control(headless=True, **deps):
    sim_device = 'cuda:0'
    Cfg = configure_env()

    PPO_Args.entropy_coef = 0.005
    RunnerArgs.num_steps_per_env = 48
    RunnerArgs.save_video_interval = 500
    RunCfg.experiment_job_type = "release"
    RunCfg.experiment_group = "z1_position_control"

    env = Z1Robot(cfg=Cfg, sim_params=None, physics_engine=None, sim_device=sim_device, headless=headless)
    # If you use wrappers, add them here
    # env = HistoryWrapper(env, reward_scaling=1.0)

    # log the experiment parameters
    import wandb
    wandb.init(
      project="z1-position-control",
      group=RunCfg.experiment_group,
      job_type=RunCfg.experiment_job_type,
      config={
        "AC_Args": vars(AC_Args),
        "PPO_Args": vars(PPO_Args),
        "RunnerArgs": vars(RunnerArgs),
        "Cfg": vars(Cfg),
      },
    )

    runner = Runner(env, device=sim_device)
    runner.learn(num_learning_iterations=10000, init_at_random_ep_len=True, eval_freq=100)

if __name__ == '__main__':
    from pathlib import Path
    stem = Path(__file__).stem
    train_z1_position_control(headless=True)
