import numpy as np
import cv2
import robosuite as suite

def evaluate_twoarmlift(n_episodes=20, horizon=500):
    """Evaluate TwoArmLift with Baxter bimanual robot for reproducibility"""
    
    results = {
        'episode_rewards': [],
        'success_rate': [],
        'avg_reward': 0.0,
        'avg_success': 0.0
    }
    
    for ep in range(n_episodes):
        env = suite.make(
            env_name="TwoArmLift",
            robots="Aloha",                        # Bimanual robot (Aloha requires robosuite_models)
            env_configuration="single-robot",       # Use bimanual robot
            has_renderer=False,                    # Disable live viewer (macOS compat)
            has_offscreen_renderer=True,           # Required for camera observations
            use_camera_obs=True,                   # Enable camera observations
            camera_names=["agentview", "robot0_wrist_cam_right", "robot0_wrist_cam_left"],
            camera_heights=[512, 512, 512],  # Different heights per camera
            camera_widths=[1586, 512, 512],  # agentview: 1586 for 128° horizontal FOV, others: 512
            camera_depths=False,                   # Set to True for RGB-D
            use_object_obs=True,
            reward_shaping=True,
            horizon=horizon,
            seed=ep  # deterministic test cases
        )

        obs = env.reset()
        
        # Adjust agentview camera: Set to 67° vertical FOV (128° horizontal with aspect ratio 3.098)
        agentview_cam_id = env.sim.model.camera_name2id("agentview")
        # Move camera further back (increase x position) - original is around 0.5, move to 1.2
        env.sim.model.cam_pos[agentview_cam_id][0] = 1.2  # x position (further back)
        env.sim.model.cam_pos[agentview_cam_id][2] = 1.5  # z position (slightly higher)
        # Set vertical FOV to 67 degrees (horizontal will be 128° with width=1586, height=512)
        env.sim.model.cam_fovy[agentview_cam_id] = 67.0
        env.sim.forward()  # Update simulation
        
        episode_reward = 0
        
        for step in range(horizon):
            # Get camera images for VLA policy
            agentview_img = obs["agentview_image"]
            right_wrist_img = obs["robot0_wrist_cam_right_image"]
            left_wrist_img = obs["robot0_wrist_cam_left_image"]
            
            # TODO: Replace with VLA policy server call
            # action = vla_policy.predict(agentview_img, right_wrist_img, left_wrist_img)
            action = np.random.randn(*env.action_spec[0].shape) * 0.1
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Display with OpenCV (BGR format, flip vertically)
            cv2.imshow("agentview", cv2.cvtColor(agentview_img[::-1], cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # Check success using internal method
        success = env._check_success()
        
        results['episode_rewards'].append(episode_reward)
        results['success_rate'].append(int(success))
        
        print(f"Episode {ep+1}/{n_episodes}: Reward={episode_reward:.2f}, Success={success}")
        
        env.close()
        cv2.destroyAllWindows()
    
    results['avg_reward'] = np.mean(results['episode_rewards'])
    results['avg_success'] = np.mean(results['success_rate'])
    
    print(f"\n{'='*50}")
    print(f"Final Evaluation Results:")
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print(f"Success Rate: {results['avg_success']*100:.1f}%")
    print(f"{'='*50}")
    
    return results

if __name__ == "__main__":
    evaluate_twoarmlift(n_episodes=20, horizon=500)