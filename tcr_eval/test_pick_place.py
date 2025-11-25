"""
TwoArmLift evaluation with Pi0.5 policy via openpi WebSocket client.

This script evaluates the TwoArmLift task using a bimanual ALOHA robot,
connecting to a remote Pi0.5 policy server via WebSocket.

IMPORTANT: Joint Ordering
    - OpenPI ALOHA expects: [left_arm(6), left_grip(1), right_arm(6), right_grip(1)]
    - Robosuite bimanual robots typically use: [right_arm, left_arm] ordering
    - This script SWAPS the ordering in make_aloha_obs() to match openpi's expectation
    - If actions are mirrored or robot behaves incorrectly, verify joint ordering!

Usage:
    # Start Pi0.5 server (on GPU machine):
    cd /path/to/openpi
    uv run scripts/serve_policy.py --env ALOHA --port 8000

    # Run evaluation (simulation machine):
    python test_pick_place.py --policy-host <server_ip> --policy-port 8000
"""

import argparse
import collections
import logging

import cv2
import numpy as np
import robosuite as suite

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy
except ImportError as e:
    raise ImportError(
        "Missing openpi_client. Install with:\n"
        "  cd /path/to/openpi/packages/openpi-client && pip install -e .\n"
        "  or: pip install websockets msgpack numpy"
    ) from e


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_aloha_obs(robosuite_obs, task_description, resize=224):
    """
    Convert robosuite observation to openpi ALOHA format.
    
    Args:
        robosuite_obs: Raw observation dict from robosuite env
        task_description: Language instruction for the task
        resize: Target image size (default: 224)
        
    Returns:
        Dictionary in openpi ALOHA format:
        {
            "state": (14,) [left_arm(6), left_grip(1), right_arm(6), right_grip(1)],
            "images": {
                "cam_high": (3, 224, 224) CHW uint8,
                "cam_left_wrist": (3, 224, 224) CHW uint8,
                "cam_right_wrist": (3, 224, 224) CHW uint8,
            },
            "prompt": str,
        }
    """
    
    def process_img(img):
        """Convert HWC uint8 -> resize -> CHW uint8."""
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, resize, resize)
        )
        return np.transpose(img, (2, 0, 1))  # HWC -> CHW
    
    # Extract joint positions
    # robosuite Aloha robot provides:
    # - robot0_joint_pos: all arm joints (12 joints for bimanual)
    # - robot0_left_gripper_qpos: left gripper joint(s)
    # - robot0_right_gripper_qpos: right gripper joint(s)
    
    joint_pos = robosuite_obs["robot0_joint_pos"]  # (12,) for bimanual
    left_gripper = robosuite_obs["robot0_left_gripper_qpos"]
    right_gripper = robosuite_obs["robot0_right_gripper_qpos"]
    
    # CRITICAL: Joint ordering
    # - OpenPI ALOHA expects: [left_arm(6), left_grip(1), right_arm(6), right_grip(1)]
    # - Robosuite bimanual robots (Baxter, Tiago) use: [right_arm, left_arm] ordering
    # - ASSUMPTION: Aloha follows same convention: [right_arm(6), left_arm(6)]
    # - Therefore, we need to SWAP the order when constructing the state
    state = np.concatenate([
        joint_pos[6:12],         # left arm joints (from robosuite right=0:6, left=6:12)
        left_gripper[:1],        # left gripper (take first joint)
        joint_pos[:6],           # right arm joints
        right_gripper[:1],       # right gripper (take first joint)
    ]).astype(np.float32)
    
    return {
        "state": state,
        "images": {
            "cam_high": process_img(robosuite_obs["agentview_image"]),
            "cam_left_wrist": process_img(robosuite_obs["robot0_wrist_cam_left_image"]),
            "cam_right_wrist": process_img(robosuite_obs["robot0_wrist_cam_right_image"]),
        },
        "prompt": task_description,
    }


def evaluate_twoarmlift(
    n_episodes: int = 20,
    horizon: int = 500,
    policy_host: str = "localhost",
    policy_port: int = 8000,
    task_description: str = "Lift the pot using both arms",
    use_policy: bool = True,
    show_images: bool = True,
    replan_steps: int = 5,
):
    """
    Evaluate TwoArmLift with Pi0.5 ALOHA policy.
    
    Args:
        n_episodes: Number of evaluation episodes
        horizon: Maximum steps per episode
        policy_host: Pi0.5 policy server hostname
        policy_port: Pi0.5 policy server port
        task_description: Language instruction for the task
        use_policy: If True, use policy server; if False, use random actions
        show_images: If True, display camera images with OpenCV
        replan_steps: Number of actions to execute before re-inference
    """
    results = {
        'episode_rewards': [],
        'success_rate': [],
        'avg_reward': 0.0,
        'avg_success': 0.0
    }
    
    # Initialize policy client if using server
    policy = None
    if use_policy:
        logger.info(f"Connecting to Pi0.5 policy server at {policy_host}:{policy_port}...")
        policy = websocket_client_policy.WebsocketClientPolicy(host=policy_host, port=policy_port)
        logger.info(f"Connected! Server metadata: {policy.get_server_metadata()}")
    
    for ep in range(n_episodes):
        env = suite.make(
            env_name="TwoArmLift",
            robots="Aloha",                        # Bimanual robot (requires robosuite_models)
            env_configuration="single-robot",       # Use bimanual robot
            has_renderer=False,                    # Disable live viewer (macOS compat)
            has_offscreen_renderer=True,           # Required for camera observations
            use_camera_obs=True,                   # Enable camera observations
            camera_names=["agentview", "robot0_wrist_cam_right", "robot0_wrist_cam_left"],
            camera_heights=[512, 512, 512],
            camera_widths=[1586, 512, 512],        # agentview: 1586 for wide FOV
            camera_depths=False,                   # Set to True for RGB-D
            use_object_obs=True,
            reward_shaping=True,
            horizon=horizon,
            seed=ep
        )

        obs = env.reset()
        
        # Adjust agentview camera: Set to 67° vertical FOV
        agentview_cam_id = env.sim.model.camera_name2id("agentview")
        env.sim.model.cam_pos[agentview_cam_id][0] = 1.2  # x position (further back)
        env.sim.model.cam_pos[agentview_cam_id][2] = 1.5  # z position (slightly higher)
        env.sim.model.cam_fovy[agentview_cam_id] = 67.0   # vertical FOV
        env.sim.forward()
        
        episode_reward = 0
        action_plan = collections.deque()  # Action buffer for chunked execution
        
        logger.info(f"Episode {ep+1}/{n_episodes}: Starting...")
        
        for step in range(horizon):
            # Get action from policy server or use random
            if policy is not None:
                try:
                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        obs_dict = make_aloha_obs(obs, task_description)
                        action_response = policy.infer(obs_dict)
                        action_chunk = action_response["actions"]  # (chunk_size, 14)
                        
                        # Take only replan_steps actions from the chunk
                        num_actions = min(replan_steps, len(action_chunk))
                        action_plan.extend(action_chunk[:num_actions])
                        
                        if step == 0:
                            logger.info(f"  Received action chunk: shape={action_chunk.shape}")
                    
                    action = action_plan.popleft()  # (14,) joint-space action
                    
                except Exception as e:
                    logger.error(f"  Policy error at step {step}: {e}")
                    action = np.random.randn(*env.action_spec[0].shape) * 0.1
            else:
                action = np.random.randn(*env.action_spec[0].shape) * 0.1
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Display with OpenCV (BGR format, flip vertically)
            if show_images:
                agentview_img = obs["agentview_image"]
                cv2.imshow("agentview", cv2.cvtColor(agentview_img[::-1], cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("  User pressed 'q', stopping episode...")
                    break
            
            if done:
                logger.info(f"  Episode completed at step {step+1}")
                break
        
        # Check success using internal method
        success = env._check_success()
        
        results['episode_rewards'].append(episode_reward)
        results['success_rate'].append(int(success))
        
        logger.info(
            f"Episode {ep+1}/{n_episodes}: Reward={episode_reward:.2f}, "
            f"Success={success}, Steps={step+1}"
        )
        
        env.close()
        if show_images:
            cv2.destroyAllWindows()
    
    results['avg_reward'] = np.mean(results['episode_rewards'])
    results['avg_success'] = np.mean(results['success_rate'])
    
    print(f"\n{'='*50}")
    print(f"Final Evaluation Results:")
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print(f"Success Rate: {results['avg_success']*100:.1f}%")
    print(f"{'='*50}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TwoArmLift with Pi0.5 policy server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With Pi0.5 server running on localhost
  python test_pick_place.py --n-episodes 10
  
  # With Pi0.5 server on remote GPU machine
  python test_pick_place.py --policy-host 192.168.1.100 --n-episodes 20
  
  # Without policy (random actions, for testing)
  python test_pick_place.py --no-policy --n-episodes 5
        """
    )
    parser.add_argument("--n-episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--horizon", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--policy-host", type=str, default="localhost", help="Policy server host")
    parser.add_argument("--policy-port", type=int, default=8000, help="Policy server port")
    parser.add_argument(
        "--task-description", type=str, default="Lift the pot using both arms",
        help="Language instruction for the task"
    )
    parser.add_argument("--no-policy", action="store_true", help="Use random actions instead of policy")
    parser.add_argument("--no-display", action="store_true", help="Disable image display")
    parser.add_argument("--replan-steps", type=int, default=5, help="Actions to execute before re-inference")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_twoarmlift(
        n_episodes=args.n_episodes,
        horizon=args.horizon,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        task_description=args.task_description,
        use_policy=not args.no_policy,
        show_images=not args.no_display,
        replan_steps=args.replan_steps,
    )
