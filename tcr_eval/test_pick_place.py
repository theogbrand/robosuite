import numpy as np
import robosuite as suite

def evaluate_pickplace(n_episodes=20, horizon=500):
    """Evaluate PickPlace with fixed seeds for reproducibility"""
    
    results = {
        'episode_rewards': [],
        'success_rate': [],
        'avg_reward': 0.0,
        'avg_success': 0.0
    }
    
    for ep in range(n_episodes):
        env = suite.make(
            env_name="PickPlace",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=True,
            horizon=horizon,
            seed=ep  # deterministic test cases
        )
        
        obs = env.reset()
        episode_reward = 0
        
        for step in range(horizon):
            # Replace with your policy
            action = np.random.randn(*env.action_spec[0].shape) * 0.1
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
        # Check success using internal method
        success = env._check_success()
        
        results['episode_rewards'].append(episode_reward)
        results['success_rate'].append(int(success))
        
        print(f"Episode {ep+1}/{n_episodes}: Reward={episode_reward:.2f}, Success={success}")
        
        env.close()
    
    results['avg_reward'] = np.mean(results['episode_rewards'])
    results['avg_success'] = np.mean(results['success_rate'])
    
    print(f"\n{'='*50}")
    print(f"Final Evaluation Results:")
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print(f"Success Rate: {results['avg_success']*100:.1f}%")
    print(f"{'='*50}")
    
    return results

if __name__ == "__main__":
    evaluate_pickplace(n_episodes=20, horizon=500)