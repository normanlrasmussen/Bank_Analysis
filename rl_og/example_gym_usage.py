"""
Example usage of the Bank Gymnasium environment with existing players.

This demonstrates how to use the Gymnasium environment with the existing
player classes from player.py.
"""
from bank_gym import BankEnv
from players import ThersholdPlayer, GreedyPlayer, GreedyPlayerK, SesquaGreedyPlayer


def example_basic_usage():
    """Basic example of using the Gymnasium environment."""
    print("Example: Basic Gymnasium Environment Usage")
    print("=" * 60)
    
    # Create opponents using existing player classes
    opponents = [
        ThersholdPlayer(threshold=50),
        GreedyPlayer()
    ]
    
    # Create the Gymnasium environment
    env = BankEnv(
        rounds=10,
        opponents=opponents,
        verbose=False,
        render_mode="human"  # Set to None to disable rendering
    )
    
    # Reset the environment
    obs, info = env.reset(seed=42)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    print()
    
    # Simple policy: bank if score > 60
    terminated = False
    truncated = False
    step_count = 0
    total_reward = 0
    
    while not (terminated or truncated) and step_count < 1000:
        # Get current state from info
        current_score = info.get("current_score", 0)
        
        # Simple policy: bank if score > 60
        action = 1 if current_score > 60 else 0  # 1 = bank, 0 = roll
        
        # Take a step
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        # Render (if enabled)
        if step_count % 10 == 0:
            env.render()
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode finished!")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final scores: {info.get('final_scores', info.get('player_scores', []))}")
    
    env.close()


def example_with_rl_algorithm():
    """
    Example showing how the environment can be used with RL algorithms.
    This is a template - you would use an actual RL library like stable-baselines3.
    """
    print("\n" + "=" * 60)
    print("Example: Using with RL Algorithms (Template)")
    print("=" * 60)
    
    # Create environment
    opponents = [ThersholdPlayer(threshold=40)]
    env = BankEnv(rounds=5, opponents=opponents)
    
    # Example: With stable-baselines3 (uncomment if you have it installed)
    # from stable_baselines3 import PPO
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=10000)
    # model.save("bank_ppo")
    
    print("Environment is ready for RL training!")
    print("You can use it with libraries like:")
    print("  - stable-baselines3")
    print("  - ray[rllib]")
    print("  - CleanRL")
    print("  - And other Gymnasium-compatible RL libraries")
    
    env.close()


if __name__ == "__main__":
    example_basic_usage()
    example_with_rl_algorithm()

