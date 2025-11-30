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
    experiment_job_type = "release"


def configure_env():
    from b1_gym.envs.base.legged_robot_config import Cfg
    config_z1(Cfg)

    # Z1 has no leg DOFs, so avoid leg gains errors
    Cfg.commands.p_gains_legs = []
    Cfg.commands.d_gains_legs = []

    # Z1-specific environment settings
    Cfg.env.num_envs = 1024*4
    Cfg.env.num_actions = 7  # 6 arm joints + 1 gripper
    # Observation size: OrientationSensor(3) + JointPositionSensor(7) + JointVelocitySensor(7) + ActionSensor(7) + ClockSensor(4) = 28
    Cfg.env.num_scalar_observations = 28
    Cfg.env.num_observations = 28
    Cfg.env.episode_length_s = 30
    Cfg.sim.physx.max_gpu_contact_pairs = 2 ** 18  # Reduced for memory
    Cfg.robot.name = "z1"

    # Commands configuration for Z1 (full pose control)
    # Command indices: [0]=radius, [1]=pos_pitch, [2]=pos_yaw, [3]=timing, [4]=roll, [5]=ori_pitch, [6]=ori_yaw
    Cfg.commands.num_commands = 7
    Cfg.commands.resampling_time = 10.0
    Cfg.commands.command_curriculum = False
    Cfg.commands.distributional_commands = False
    Cfg.commands.control_only_z1 = False  # We're training pure Z1, not B1+Z1
    
    # EE spherical position command ranges (relative to arm base)
    Cfg.commands.ee_sphe_radius = [0.3, 0.7]
    Cfg.commands.ee_sphe_pitch = [-np.pi/3, np.pi/3]
    Cfg.commands.ee_sphe_yaw = [-np.pi/2, np.pi/2]
    Cfg.commands.limit_ee_sphe_radius = [0.3, 0.7]
    Cfg.commands.limit_ee_sphe_pitch = [-np.pi/3, np.pi/3]
    Cfg.commands.limit_ee_sphe_yaw = [-np.pi/2, np.pi/2]
    
    # EE orientation command ranges (roll, pitch, yaw in radians)
    Cfg.commands.ee_ori_roll = [-np.pi/4, np.pi/4]
    Cfg.commands.ee_ori_pitch = [-np.pi/3, np.pi/3]
    Cfg.commands.ee_ori_yaw = [-np.pi/2, np.pi/2]
    Cfg.commands.limit_ee_ori_roll = [-np.pi/4, np.pi/4]
    Cfg.commands.limit_ee_ori_pitch = [-np.pi/3, np.pi/3]
    Cfg.commands.limit_ee_ori_yaw = [-np.pi/2, np.pi/2]
    
    # Timing command (not actively used but part of command vector)
    Cfg.commands.ee_timing = [0.0, 0.0]
    Cfg.commands.limit_ee_timing = [0.0, 0.0]

    # Sensors for Z1 arm
    Cfg.sensors.sensor_names = [
        "OrientationSensor",    # size 3: projected gravity
        "JointPositionSensor",  # size 7: arm joint positions
        "JointVelocitySensor",  # size 7: arm joint velocities
        "ActionSensor",         # size 7: current actions
        "ClockSensor",          # size 4: phase clock
    ]
    Cfg.sensors.sensor_args = {
        "OrientationSensor": {},
        "JointPositionSensor": {},
        "JointVelocitySensor": {},
        "ActionSensor": {},
        "ClockSensor": {},
    }
    
    # Privileged observations for adaptation
    Cfg.sensors.privileged_sensor_names = [
        "JointDynamicsSensor",       # size 3: motor dynamics
        "EeGripperPositionSensor",   # size 3: EE position
    ]
    Cfg.sensors.privileged_sensor_args = {
        "JointDynamicsSensor": {},
        "EeGripperPositionSensor": {},
    }
    Cfg.env.num_privileged_obs = 6
    
    # Adaptation module settings
    AC_Args.adaptation_labels = ["dynamics_loss", "gripper_pos_loss"]
    AC_Args.adaptation_dims = [3, 3]
    AC_Args.adaptation_weights = [1, 10]
    AC_Args.init_noise_std = 1.0

    Cfg.env.num_observation_history = 5  # Reduced from 10 for memory
    Cfg.env.history_frame_skip = 1

    # Reward container for Z1
    Cfg.rewards.reward_container_name = "Z1ArmBaseFrameRewards"
    
    # Reward settings
    Cfg.rewards.only_positive_rewards = True
    Cfg.rewards.only_positive_rewards_ji22_style = False
    Cfg.rewards.total_rew_scale = 0.2
    Cfg.rewards.soft_dof_pos_limit = 0.9
    Cfg.rewards.soft_torque_limit_arm = 1.0
    Cfg.rewards.base_height_target = 0.70
    
    # Termination conditions
    Cfg.rewards.use_terminal_body_height = False
    Cfg.rewards.use_terminal_roll_pitch = False
    Cfg.rewards.use_terminal_foot_height = False
    Cfg.rewards.use_terminal_torque_arm_limits = False

    ######################
    ######## ARM #########
    ######################
    # Position and orientation tracking rewards
    Cfg.reward_scales.manip_pos_tracking = 3.0
    Cfg.reward_scales.manip_ori_tracking = 0.0 # disabled
    
    # Torque and dynamics penalties
    Cfg.reward_scales.torque_limits_arm = -0.005
    Cfg.reward_scales.torques = -1e-5
    Cfg.reward_scales.dof_vel_arm = -0.003
    Cfg.reward_scales.dof_acc_arm = -3e-8
    
    # Action smoothness penalties
    Cfg.reward_scales.action_rate_arm = 0.0
    Cfg.reward_scales.action_smoothness_1_arm = -0.01
    Cfg.reward_scales.action_smoothness_2_arm = -0.01
    
    # Joint limit penalties
    Cfg.reward_scales.dof_pos_limits_arm = -10.0
    Cfg.reward_scales.dof_pos = 0.0  # disabled
    
    # Survival reward
    Cfg.reward_scales.survival = 2.0
    
    # Collision penalty
    Cfg.reward_scales.collision = -5.0
    
    # EE motion penalties
    Cfg.reward_scales.ee_velocity = 0.0
    Cfg.reward_scales.ee_acceleration = 0.0
    
    # Regularization rewards
    Cfg.reward_scales.joint_regularization = 0.0
    Cfg.reward_scales.manipulability = 0.0
    
    # Orientation/base rewards (minimal for arm)
    Cfg.reward_scales.orientation = 0.0
    Cfg.reward_scales.base_height = 0.0
    Cfg.reward_scales.termination = 0.0
    
    # Disable leg-specific rewards that don't exist in Z1ArmBaseFrameRewards
    Cfg.reward_scales.tracking_lin_vel = 0.0
    Cfg.reward_scales.tracking_ang_vel = 0.0
    Cfg.reward_scales.lin_vel_z = 0.0
    Cfg.reward_scales.ang_vel_xy = 0.0
    Cfg.reward_scales.dof_acc_leg = 0.0
    Cfg.reward_scales.feet_air_time = 0.0
    Cfg.reward_scales.action_rate_leg = 0.0
    Cfg.reward_scales.action_rate = 0.0
    Cfg.reward_scales.dof_pos_limits = 0.0  # Use dof_pos_limits_arm instead

    # Domain randomization
    Cfg.domain_rand.rand_interval_s = 4
    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.domain_rand.randomize_rigids_after_start = False
    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [0.6, 5.0]
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.restitution_range = [0.0, 0.4]
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [-0.5, 1.0]
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.gravity_range = [-0.01, 0.01]
    Cfg.domain_rand.randomize_com_displacement = True
    Cfg.domain_rand.com_displacement_range = [-0.05, 0.05]
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.ground_friction_range = [0.0, 0.01]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_tile_roughness = False
    Cfg.domain_rand.tile_roughness_range = [0.0, 0.1]

    # Position control gains
    Cfg.commands.p_gains_arm = [60.0, 90.0, 60.0, 60.0, 45.0, 30.0, 60.0]
    Cfg.commands.d_gains_arm = [1.5, 3.0, 1.5, 1.5, 1.5, 1.5, 1.5]
    
    # Control settings
    Cfg.control.decimation = 4
    Cfg.control.control_type = 'P'
    Cfg.control.action_scale = 0.25
    Cfg.control.arm_scale_reduction = 2.0
    
    # Normalization
    Cfg.normalization.clip_actions = 10.0

    # Terrain settings (simple plane for arm training - low memory)
    Cfg.terrain.mesh_type = 'plane'  # Use plane instead of trimesh to save memory
    Cfg.terrain.terrain_noise_magnitude = 0.0
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.border_size = 5
    Cfg.terrain.num_cols = 1
    Cfg.terrain.num_rows = 1
    Cfg.terrain.terrain_width = 5.0
    Cfg.terrain.terrain_length = 5.0
    Cfg.terrain.x_init_range = 0.5
    Cfg.terrain.y_init_range = 0.5
    Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    Cfg.terrain.curriculum = False
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 4

    # Asset settings
    Cfg.asset.fix_base_link = True  # Z1 arm is typically fixed-base
    Cfg.asset.penalize_contacts_on = ["link02", "link03", "link06"]
    Cfg.asset.terminate_after_contacts_on = []
    Cfg.asset.self_collisions = 0  # enable self-collision

    # Initial state: place Z1 base at world origin
    Cfg.init_state.pos = [0.0, 0.0, 0.0]
    Cfg.init_state.default_joint_angles = {
        'joint1': 0.0,
        'joint2': 1.5,
        'joint3': -1.5,
        'joint4': -0.54,
        'joint5': 0.0,
        'joint6': 0.0,
        'jointGripper': 0.0,
    }

    # Viewer settings
    Cfg.viewer.follow_robot = False
    Cfg.env.recording_width_px = 180
    Cfg.env.recording_height_px = 120

    return Cfg

def train_z1_position_control(headless=True, **deps):

    sim_device = 'cuda:0'
    Cfg = configure_env()

    # Create sim_params and select physics engine
    from isaacgym import gymapi
    sim_params = gymapi.SimParams()
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim_params.dt = 0.005  # 200 Hz simulation
    sim_params.substeps = 1
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # PhysX parameters
    sim_params.physx.num_threads = 10
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.bounce_threshold_velocity = 0.5
    sim_params.physx.max_depenetration_velocity = 1.0
    sim_params.physx.max_gpu_contact_pairs = 2 ** 16  # Minimal for arm-only
    sim_params.physx.default_buffer_size_multiplier = 1  # Minimal buffer
    sim_params.physx.contact_collection = gymapi.CC_NEVER
    
    physics_engine = gymapi.SIM_PHYSX

    # PPO and Runner settings
    PPO_Args.entropy_coef = 0.01  # Slightly higher for more exploration
    PPO_Args.learning_rate = 3e-4  # Lower LR for more stable learning
    PPO_Args.num_learning_epochs = 4  # Fewer epochs per update
    PPO_Args.gamma = 0.99
    PPO_Args.lam = 0.95
    
    RunnerArgs.num_steps_per_env = 48
    RunnerArgs.save_video_interval = 500
    
    RunCfg.experiment_job_type = "release"
    RunCfg.experiment_group = "z1_position_control"

    # Create the Z1 environment
    env = Z1Robot(
        cfg=Cfg, 
        sim_params=sim_params, 
        physics_engine=physics_engine, 
        sim_device=sim_device, 
        headless=headless
    )
    
    # Add history wrapper for temporal information (required by Runner)
    from b1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    env = HistoryWrapper(env, reward_scaling=1.0)

    # Initialize wandb logging
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

    # Create runner and start training
    runner = Runner(env, device=sim_device)
    runner.learn(num_learning_iterations=10000, init_at_random_ep_len=True, eval_freq=100)


if __name__ == '__main__':
    from pathlib import Path
    stem = Path(__file__).stem
    train_z1_position_control(headless=True)
