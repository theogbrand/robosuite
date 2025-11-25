"""
TwoArmLift evaluation with Pi0.5 policy via Modal-deployed OpenPI server.

This script evaluates the TwoArmLift task using a bimanual ALOHA robot,
connecting to a Modal-deployed Pi0.5 policy server via WebSocket.

IMPORTANT: Joint Ordering
    - OpenPI ALOHA expects: [left_arm(6), left_grip(1), right_arm(6), right_grip(1)]
    - Robosuite bimanual robots typically use: [right_arm, left_arm] ordering
    - This script SWAPS the ordering in make_aloha_obs() to match openpi's expectation
    - If actions are mirrored or robot behaves incorrectly, verify joint ordering!

Usage:
    # With default Modal endpoint and model:
    python test_pick_place.py --n-episodes 10

    # Override Modal endpoint or model config:
    python test_pick_place.py --modal-url <url> --modal-hf-repo-id <repo> --n-episodes 20
"""

import argparse
import collections
import logging

import cv2
import numpy as np
import robosuite as suite

from utils.modal_client import ModalClientPolicy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Modal policy server configuration
MODAL_URL = "https://griffin-labs--openpi-policy-server-endpoint.modal.run/"
MODAL_HF_REPO_ID = "griffinlabs/pi05_412ep_pytorch"
MODAL_FOLDER_PATH = "pi05_tcr_full_finetune_pytorch/pi05_412ep/20000"
MODAL_CONFIG_NAME = "pi05_tcr_full_finetune_pytorch"
MODAL_PROMPT = "pick up the object"
MODAL_DATASET_REPO_ID = "griffinlabs/tcr-data"
MODAL_STATS_JSON_PATH = "./norm_stats.json"


def make_aloha_obs(robosuite_obs, task_description, resize=224):
    """
    Convert robosuite observation to openpi TCR format.
    
    Args:
        robosuite_obs: Raw observation dict from robosuite env
        task_description: Language instruction for the task
        resize: Target image size (default: 224)
        
    Returns:
        Dictionary in openpi TCR format:
        {
            "state": (14,) [left_arm(6), left_grip(1), right_arm(6), right_grip(1)],
            "images": {
                "top": (224, 224, 3) HWC uint8,
                "front": (224, 224, 3) HWC uint8,
                "left": (224, 224, 3) HWC uint8,
                "right": (224, 224, 3) HWC uint8,
            },
            "prompt": str,
        }
    """
    
    def process_img(img):
        """Convert HWC uint8 -> resize -> HWC uint8 (keep HWC for TCR)."""
        # Resize with padding to maintain aspect ratio
        h, w = img.shape[:2]
        scale = min(resize/h, resize/w)
        new_h, new_w = int(h*scale), int(w*scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        padded = np.zeros((resize, resize, 3), dtype=np.uint8)
        y_offset = (resize - new_h) // 2
        x_offset = (resize - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded  # Keep HWC format for TCR
    
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
    
    logging.info(f"state: {state.shape}")
    logging.info(f"images (original): {robosuite_obs['agentview_image'].shape}")
    logging.info(f"images (original): {robosuite_obs['robot0_wrist_cam_left_image'].shape}")
    logging.info(f"images (original): {robosuite_obs['robot0_wrist_cam_right_image'].shape}")
    logging.info(f"prompt: {task_description}")
    
    # TCR camera mapping (4 cameras required):
    # - top: overhead view (agentview)
    # - front: front view (use agentview as fallback since no dedicated front camera)
    # - left: left wrist camera
    # - right: right wrist camera
    return {
        "state": state,
        "images": {
            "top": process_img(robosuite_obs["agentview_image"]),
            "front": process_img(robosuite_obs["agentview_image"]),  # Duplicate as fallback
            "left": process_img(robosuite_obs["robot0_wrist_cam_left_image"]),
            "right": process_img(robosuite_obs["robot0_wrist_cam_right_image"]),
        },
        "prompt": task_description,
    }


def evaluate_twoarmlift(
    n_episodes: int = 20,
    horizon: int = 500,
    modal_url: str = MODAL_URL,
    modal_hf_repo_id: str = MODAL_HF_REPO_ID,
    modal_folder_path: str = MODAL_FOLDER_PATH,
    modal_config_name: str = MODAL_CONFIG_NAME,
    task_description: str = "Lift the pot using both arms",
    show_images: bool = True,
    replan_steps: int = 5,
):
    """
    Evaluate TwoArmLift with Pi0.5 ALOHA policy via Modal endpoint.
    
    Args:
        n_episodes: Number of evaluation episodes
        horizon: Maximum steps per episode
        modal_url: Modal endpoint URL (HTTPS, will be converted to WSS)
        modal_hf_repo_id: HuggingFace repo ID for the model
        modal_folder_path: Path to checkpoint folder within the HF repo
        modal_config_name: Config name to use for loading the policy
        task_description: Language instruction for the task
        show_images: If True, display camera images with OpenCV
        replan_steps: Number of actions to execute before re-inference
    """
    results = {
        'episode_rewards': [],
        'success_rate': [],
        'avg_reward': 0.0,
        'avg_success': 0.0
    }
    
    # Convert HTTP URL to WebSocket URL
    ws_url = modal_url.replace("https://", "wss://").replace("http://", "ws://")
    if not ws_url.endswith("/ws"):
        ws_url = ws_url.rstrip("/") + "/ws"
    
    logger.info(f"Connecting to Modal policy at {ws_url}...")
    logger.info(f"Model: {modal_config_name} from {modal_hf_repo_id}/{modal_folder_path}")
    
    policy = ModalClientPolicy(
        url=ws_url,
        hf_repo_id=modal_hf_repo_id,
        folder_path=modal_folder_path,
        config_name=modal_config_name,
        prompt=MODAL_PROMPT,
        dataset_repo_id=MODAL_DATASET_REPO_ID,
        stats_json_path=MODAL_STATS_JSON_PATH,
    )
    logger.info(f"Connected! Policy metadata: {policy.get_policy_metadata()}")
    
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
        
        # Adjust agentview camera: More top-down view
        agentview_cam_id = env.sim.model.camera_name2id("agentview")
        env.sim.model.cam_pos[agentview_cam_id][0] = 0.5   # x: slightly forward from center
        env.sim.model.cam_pos[agentview_cam_id][1] = 0.0   # y: centered
        env.sim.model.cam_pos[agentview_cam_id][2] = 1.4   # z: moderate height
        env.sim.model.cam_fovy[agentview_cam_id] = 50.0    # moderate FOV
        # Don't change quaternion - use default orientation
        env.sim.forward()
        
        episode_reward = 0
        action_plan = collections.deque()  # Action buffer for chunked execution
        
        logger.info(f"Episode {ep+1}/{n_episodes}: Starting...")
        
        for step in range(horizon):
            # Get action from policy server
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
        description="Evaluate TwoArmLift with Modal-deployed Pi0.5 policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With default Modal endpoint and model
  python test_pick_place.py --n-episodes 10
  
  # Override Modal endpoint or model config
  python test_pick_place.py --modal-url <url> --modal-hf-repo-id <repo> --n-episodes 20
        """
    )
    parser.add_argument("--n-episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--horizon", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--modal-url", type=str, default=MODAL_URL, help=f"Modal endpoint URL (default: {MODAL_URL})")
    parser.add_argument("--modal-hf-repo-id", type=str, default=MODAL_HF_REPO_ID, help=f"HuggingFace repo ID (default: {MODAL_HF_REPO_ID})")
    parser.add_argument("--modal-folder-path", type=str, default=MODAL_FOLDER_PATH, help="Checkpoint folder path in HF repo")
    parser.add_argument("--modal-config-name", type=str, default=MODAL_CONFIG_NAME, help="Config name for policy loading")
    parser.add_argument(
        "--task-description", type=str, default="Lift the pot using both arms",
        help="Language instruction for the task"
    )
    parser.add_argument("--no-display", action="store_true", help="Disable image display")
    parser.add_argument("--replan-steps", type=int, default=5, help="Actions to execute before re-inference")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_twoarmlift(
        n_episodes=args.n_episodes,
        horizon=args.horizon,
        modal_url=args.modal_url,
        modal_hf_repo_id=args.modal_hf_repo_id,
        modal_folder_path=args.modal_folder_path,
        modal_config_name=args.modal_config_name,
        task_description=args.task_description,
        show_images=not args.no_display,
        replan_steps=args.replan_steps,
    )
