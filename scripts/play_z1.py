"""
Play script for trained Z1 arm-only policy.

Interactive mode: Enter EE target as 'x y z roll pitch yaw' (position + orientation).
The policy was trained with spherical coordinates internally.
Press Ctrl+C to exit.
"""

import os
import sys
import isaacgym
assert isaacgym
import torch
import argparse
import threading
import math
from isaacgym import gymapi

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from b1_gym.envs.base.legged_robot_config import Cfg
from b1_gym.envs.z1.z1_config import config_z1
from b1_gym.envs.z1.z1_robot import Z1Robot
from b1_gym.envs.wrappers.history_wrapper import HistoryWrapper
from b1_gym_learn.ppo_cse.actor_critic import ActorCritic, AC_Args


DEFAULT_CHECKPOINT = "scripts/checkpoints/ac_weights_100.pt"


def cartesian_to_spherical(x, y, z):
    """Convert Cartesian (x, y, z) to spherical (radius, pitch, yaw)."""
    radius = math.sqrt(x*x + y*y + z*z)
    if radius < 1e-6:
        return 0.5, 0.0, 0.0  # Default
    pitch = math.asin(-z / radius)  # Elevation angle (negative z is up in this convention)
    yaw = math.atan2(y, x)
    return radius, pitch, yaw


def spherical_to_cartesian(radius, pitch, yaw):
    """Convert spherical (radius, pitch, yaw) to Cartesian (x, y, z)."""
    x = radius * math.cos(pitch) * math.cos(yaw)
    y = radius * math.cos(pitch) * math.sin(yaw)
    z = -radius * math.sin(pitch)  # negative because pitch down = positive z
    return x, y, z


def draw_target_sphere(gym, viewer, envs, target_pos, color=(1.0, 0.0, 0.0)):
    """
    Draw a sphere at the target position to visualize the command.
    
    Args:
        gym: IsaacGym instance
        viewer: Viewer handle
        envs: List of environment handles
        target_pos: (x, y, z) target position
        color: RGB tuple (default: red)
    """
    if viewer is None:
        return
    
    # Clear previous lines
    gym.clear_lines(viewer)
    
    # Draw sphere as a set of lines (cross pattern)
    sphere_size = 0.03  # 3cm radius visualization
    x, y, z = target_pos
    
    # Create line vertices for a 3D cross at target
    lines = [
        # X axis (red)
        [x - sphere_size, y, z, x + sphere_size, y, z],
        # Y axis (green) 
        [x, y - sphere_size, z, x, y + sphere_size, z],
        # Z axis (blue)
        [x, y, z - sphere_size, x, y, z + sphere_size],
    ]
    
    colors = [
        [1.0, 0.0, 0.0],  # Red for X
        [0.0, 1.0, 0.0],  # Green for Y
        [0.0, 0.0, 1.0],  # Blue for Z
    ]
    
    for env in envs:
        for line, col in zip(lines, colors):
            gym.add_lines(viewer, env, 1, line, col)


def pose_to_commands(x, y, z, roll, pitch_ori, yaw_ori):
    """
    Convert full pose (position + orientation) to command format.
    
    Args:
        x, y, z: Cartesian position
        roll, pitch_ori, yaw_ori: Orientation in radians
    
    Returns:
        7 command values: [radius, pos_pitch, pos_yaw, timing, roll, ori_pitch, ori_yaw]
    """
    # Position -> spherical
    radius, pos_pitch, pos_yaw = cartesian_to_spherical(x, y, z)
    
    # Timing command (not used interactively, set to 0)
    timing = 0.0
    
    return [radius, pos_pitch, pos_yaw, timing, roll, pitch_ori, yaw_ori]


def load_policy(checkpoint_path: str, device: str = 'cuda:0'):
    """Load trained Z1 policy from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Set AC_Args to match training config
    AC_Args.adaptation_module_branch_hidden_dims = [256, 128]
    AC_Args.adaptation_dims = [3, 3]  # Total latent dim = 6
    
    # Z1 training used: num_obs=31, num_privileged_obs=6, num_obs_history=155 (5 frames × 31), num_actions=7
    actor_critic = ActorCritic(
        num_obs=31,
        num_privileged_obs=6,  # This determines adaptation module output size
        num_obs_history=155,   # 5 history frames × 31 observations
        num_actions=7
    ).to(device)
    
    actor_critic.load_state_dict(checkpoint)
    actor_critic.eval()
    print("Policy loaded!")
    
    def policy(obs):
        with torch.no_grad():
            latent = actor_critic.adaptation_module(obs["obs_history"].to(device))
            action = actor_critic.actor_body(torch.cat((obs["obs_history"].to(device), latent), dim=-1))
        return action
    
    return policy


def create_env(headless: bool = False, device: str = 'cuda:0'):
    """Create Z1 environment."""
    config_z1(Cfg)
    
    # Robot configuration - must be set to use correct robot class
    Cfg.robot.name = "z1"
    
    # Place Z1 base at world origin
    Cfg.init_state.pos = [0.0, 0.0, 0.0]
    
    Cfg.env.num_envs = 1
    Cfg.env.num_actions = 7
    Cfg.env.num_observations = 31
    Cfg.env.num_scalar_observations = 31
    Cfg.env.num_privileged_obs = 6
    Cfg.env.num_observation_history = 5  # 5 frames × 31 obs = 155 total
    Cfg.commands.num_commands = 7  # radius, pos_pitch, pos_yaw, timing, roll, ori_pitch, ori_yaw
    Cfg.commands.p_gains_legs = []
    Cfg.commands.d_gains_legs = []
    Cfg.commands.p_gains_arm = [60.0, 90.0, 60.0, 60.0, 45.0, 30.0, 60.0]
    Cfg.commands.d_gains_arm = [1.5, 3.0, 1.5, 1.5, 1.5, 1.5, 1.5]
    Cfg.control.decimation = 4
    Cfg.control.control_type = 'P'
    Cfg.control.action_scale = 0.25
    Cfg.asset.default_dof_drive_mode = 1
    Cfg.asset.fix_base_link = True  # Fix Z1 base to ground
    Cfg.rewards.reward_container_name = "Z1ArmBaseFrameRewards"
    
    # Sensors for Z1 arm (must match training)
    Cfg.sensors.sensor_names = [
        "OrientationSensor",
        "JointPositionSensor",
        "JointVelocitySensor",
        "ActionSensor",
        "ClockSensor",
        "Z1CommandSensor",
    ]
    Cfg.sensors.sensor_args = {
        "OrientationSensor": {},
        "JointPositionSensor": {},
        "JointVelocitySensor": {},
        "ActionSensor": {},
        "ClockSensor": {},
        "Z1CommandSensor": {"include_orientation": False},
    }
    Cfg.sensors.privileged_sensor_names = [
        "JointDynamicsSensor",
        "EeGripperPositionSensor",
    ]
    Cfg.sensors.privileged_sensor_args = {
        "JointDynamicsSensor": {},
        "EeGripperPositionSensor": {},
    }
    
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.terrain.mesh_type = 'plane'
    Cfg.terrain.curriculum = False
    
    sim_params = gymapi.SimParams()
    sim_params.dt = Cfg.sim.dt
    sim_params.substeps = Cfg.sim.substeps
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.num_threads = 10
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.bounce_threshold_velocity = 0.5
    sim_params.physx.max_depenetration_velocity = 1.0
    sim_params.physx.max_gpu_contact_pairs = 2**21
    sim_params.physx.default_buffer_size_multiplier = 5
    sim_params.physx.contact_collection = gymapi.ContactCollection(2)
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    
    env = Z1Robot(
        cfg=Cfg,
        sim_params=sim_params,
        physics_engine=gymapi.SIM_PHYSX,
        sim_device=device,
        headless=headless
    )
    
    return HistoryWrapper(env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play Z1 policy')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    print("Creating environment...")
    env = create_env(headless=args.headless, device=args.device)
    
    print(f"Loading policy from {args.checkpoint}...")
    policy = load_policy(args.checkpoint, device=args.device)
    
    obs = env.reset()
    
    # Initial EE pose [x=0.1, y=0, z=0.5, roll=0, pitch=0, yaw=0]
    init_x, init_y, init_z = 0.1, 0.0, 0.5
    init_roll, init_pitch_ori, init_yaw_ori = 0.0, 0.0, 0.0
    init_cmd = pose_to_commands(init_x, init_y, init_z, init_roll, init_pitch_ori, init_yaw_ori)
    print(f"Initial EE target: x={init_x}, y={init_y}, z={init_z}, roll={init_roll}, pitch={init_pitch_ori}, yaw={init_yaw_ori}")
    print(f"  -> commands: radius={init_cmd[0]:.3f}, pos_pitch={init_cmd[1]:.3f}, pos_yaw={init_cmd[2]:.3f}")
    print(f"               ori: roll={init_cmd[4]:.3f}, pitch={init_cmd[5]:.3f}, yaw={init_cmd[6]:.3f}")
    
    # Current command (shared between threads) - 7 values
    current_cmd = init_cmd
    cmd_lock = threading.Lock()
    running = True
    
    def input_thread():
        """Thread to handle user input without blocking simulation."""
        global current_cmd, running
        print("\nEnter EE target as 'x y z roll pitch yaw' (position + orientation)")
        print("  - Position (x, y, z) in meters")
        print("  - Orientation (roll, pitch, yaw) in radians")
        print("  - Can also enter just 'x y z' (orientation defaults to 0)")
        print("Type 'q' to quit\n")
        
        while running:
            try:
                user_input = input(">> ").strip()
                if user_input.lower() == 'q':
                    running = False
                    break
                
                values = user_input.split()
                if len(values) >= 3:
                    vals = [float(v) for v in values]
                    
                    # Position
                    x, y, z = vals[0], vals[1], vals[2]
                    
                    # Orientation (optional, defaults to 0)
                    if len(values) >= 6:
                        roll, pitch_ori, yaw_ori = vals[3], vals[4], vals[5]
                    else:
                        roll, pitch_ori, yaw_ori = 0.0, 0.0, 0.0
                    
                    # Convert to command format
                    new_cmd = pose_to_commands(x, y, z, roll, pitch_ori, yaw_ori)
                    
                    with cmd_lock:
                        current_cmd = new_cmd
                    
                    print(f"Pose: x={x:.3f}, y={y:.3f}, z={z:.3f}, roll={roll:.3f}, pitch={pitch_ori:.3f}, yaw={yaw_ori:.3f}")
                    print(f"  -> cmd: r={new_cmd[0]:.3f}, pos_p={new_cmd[1]:.3f}, pos_y={new_cmd[2]:.3f}, ori=({new_cmd[4]:.3f}, {new_cmd[5]:.3f}, {new_cmd[6]:.3f})")
                else:
                    print("Usage: Enter 'x y z [roll pitch yaw]' (6 values, or 3 for position only)")
            except ValueError:
                print("Invalid input. Enter numbers separated by spaces.")
            except EOFError:
                running = False
                break
    
    # Start input thread
    input_handler = threading.Thread(target=input_thread, daemon=True)
    input_handler.start()
    
    print("\nSimulation running. Enter commands to move the arm.")
    print(f"Initial command: r={current_cmd[0]:.3f}, pos=({current_cmd[1]:.3f}, {current_cmd[2]:.3f}), ori=({current_cmd[4]:.3f}, {current_cmd[5]:.3f}, {current_cmd[6]:.3f})")
    
    try:
        while running:
            # Update commands from current input (7 values)
            with cmd_lock:
                env.env.commands[:, 0] = current_cmd[0]  # radius
                env.env.commands[:, 1] = current_cmd[1]  # pos_pitch
                env.env.commands[:, 2] = current_cmd[2]  # pos_yaw
                env.env.commands[:, 3] = current_cmd[3]  # timing
                env.env.commands[:, 4] = current_cmd[4]  # roll
                env.env.commands[:, 5] = current_cmd[5]  # ori_pitch
                env.env.commands[:, 6] = current_cmd[6]  # ori_yaw
                
                # Calculate target position in Cartesian for visualization
                target_x, target_y, target_z = spherical_to_cartesian(
                    current_cmd[0], current_cmd[1], current_cmd[2]
                )
            
            # Draw target position marker
            if not args.headless and hasattr(env.env, 'viewer') and env.env.viewer is not None:
                draw_target_sphere(env.env.gym, env.env.viewer, env.env.envs, 
                                   (target_x, target_y, target_z))
            
            # Step simulation
            actions = policy(obs)
            obs, rew, done, info = env.step(actions)
            if not args.headless:
                env.render()
    except KeyboardInterrupt:
        running = False
        print("\nExiting...")