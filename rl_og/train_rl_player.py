"""
RL Training Environment for Bank Game

This script allows you to train an RL agent to play the Bank push-your-luck dice game.
All parameters are configurable at the top, and statistics are visualized at the end.
"""

import sys
from pathlib import Path

# Add parent directory to path to import game_theory module
# This ensures game_theory can be found regardless of where the script is run from
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Add project root to path (for game_theory module)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also add parent directory as fallback
parent_dir = str(project_root.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import os
import gymnasium as gym

# Try to import stable-baselines3, fall back to basic implementation if not available
try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.callbacks import CallbackList, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    raise ImportError("stable-baselines3 not available. Please install it with: pip install stable-baselines3")

# Import local modules with fallback for path issues
try:
    from bank_gym import BankEnv
    from players import ThersholdPlayer, GreedyPlayer, GreedyPlayerK, SesquaGreedyPlayer, ProbabilisticPlayer
except ModuleNotFoundError as e:
    if 'game_theory' in str(e):
        # Ensure project_root is definitely in the path and retry
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        from bank_gym import BankEnv
        from players import ThersholdPlayer, GreedyPlayer, GreedyPlayerK, SesquaGreedyPlayer, ProbabilisticPlayer
    else:
        raise


# ============================================================================
# PARAMETERS - Edit these to configure your training
# ============================================================================

# Environment Parameters
ENV_ROUNDS = 10  # Number of rounds per game
ENV_OPPONENTS = [
    ThersholdPlayer(threshold=100),
    ThersholdPlayer(threshold=200),
    ThersholdPlayer(threshold=2000),
    ProbabilisticPlayer(probability=0.2),
]  # List of opponents for the RL agent
ENV_MAX_ROUND_LENGTH = 100  # Maximum steps per round

# ============================================================================
# ALGORITHM SELECTION - Choose one of the top 3 recommended algorithms
# ============================================================================

# Algorithm to use (choose from: "PPO", "A2C", "DQN")
# All algorithms support discrete action spaces (perfect for Bank game)
TRAIN_ALGORITHM = "PPO"

# Algorithm-specific hyperparameter configurations
# Each algorithm has optimized defaults for the Bank game environment

ALGORITHM_CONFIGS = {
    # 1. PPO (Proximal Policy Optimization) - RECOMMENDED
    # Best for: Stable learning, good sample efficiency, handles discrete actions well
    # Pros: Very stable, good performance, widely used
    # Cons: Can be slower than A2C, requires more hyperparameter tuning
    "PPO": {
        "learning_rate": 1e-4,
        "n_steps": 2048,          # Steps to collect before updating
        "batch_size": 128,        # Batch size for training
        "n_epochs": 10,           # Number of epochs per update
        "gamma": 0.99,            # Discount factor
        "gae_lambda": 0.95,       # GAE lambda parameter
        "clip_range": 0.2,        # PPO clip range
        "ent_coef": 0.01,         # Entropy coefficient (encourages exploration)
        "vf_coef": 0.5,           # Value function coefficient
        "max_grad_norm": 0.5,     # Gradient clipping
    },
    
    # 2. A2C (Advantage Actor-Critic) - FAST ALTERNATIVE
    # Best for: Faster training, simpler implementation, good baseline
    # Pros: Faster than PPO, simpler, good for quick experiments
    # Cons: Less stable than PPO, may require more timesteps
    "A2C": {
        "learning_rate": 3e-4,
        "n_steps": 5,             # Steps to collect before updating (shorter rollouts)
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_rms_prop": True,     # Use RMSprop optimizer
    },
    
    # 3. DQN (Deep Q-Network) - VALUE-BASED METHOD
    # Best for: Discrete action spaces, value-based learning
    # Pros: Classic algorithm, good for discrete actions, well-understood
    # Cons: Can be sample inefficient, requires replay buffer tuning
    "DQN": {
        "learning_rate": 1e-4,
        "buffer_size": 100000,    # Replay buffer size
        "learning_starts": 1000,  # Steps before learning starts
        "batch_size": 32,         # Batch size for replay
        "tau": 1.0,               # Hard update (1.0) or soft update (<1.0)
        "gamma": 0.99,
        "train_freq": 4,          # Train every N steps
        "gradient_steps": 1,      # Gradient steps per training call
        "target_update_interval": 1000,  # Update target network every N steps
        "exploration_fraction": 0.1,     # Fraction of training for exploration
        "exploration_initial_eps": 1.0,  # Initial epsilon for epsilon-greedy
        "exploration_final_eps": 0.05,   # Final epsilon for epsilon-greedy
    },
    
}

# Get the configuration for the selected algorithm
ALGORITHM_CONFIG = ALGORITHM_CONFIGS.get(TRAIN_ALGORITHM, ALGORITHM_CONFIGS["PPO"])

# ============================================================================
# TRAINING PARAMETERS (General)
# ============================================================================

TRAIN_TOTAL_TIMESTEPS = 500000  # Total number of training steps
TRAIN_VERBOSE = 1  # Verbosity level (0, 1, or 2)
TRAIN_LOG_INTERVAL = 10  # Log progress every N episodes

# Evaluation Parameters
EVAL_N_EPISODES = 1000  # Number of episodes for post-training evaluation
EVAL_N_MATCHES_FOR_PROJECTION = [10, 50, 100, 500, 1000]  # Match counts for expected wins projection

# Training Progression Tracking
TRACK_PROGRESS_INTERVAL = 1000  # Track progress every N timesteps
PROGRESS_EVAL_EPISODES = 20  # Episodes to run for progress tracking

# Model Saving
SAVE_MODEL = True  # Whether to save the trained model
MODEL_SAVE_PATH = "RL_data/rl_bank_model_ppo_multi.zip"  # Path to save model

# Visualization Parameters
PLOT_FIGURE_SIZE = (14, 10)  # Figure size for the plot
PLOT_DPI = 100  # DPI for saving plots
SAVE_PLOT = True  # Whether to save the plot
PLOT_SAVE_PATH = "RL_data/rl_training_stats_ppo_multi.png"  # Path to save plot


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def evaluate_agent(env: BankEnv, model, n_episodes: int = 100) -> Tuple[float, float, List[int], List[float]]:
    """
    Evaluate the agent and return statistics.
    
    Returns:
        win_rate: Win rate (0-1)
        avg_score: Average score per game
        episode_results: List of win/loss/tie results (1=win, 0=tie, -1=loss)
        episode_scores: List of final scores per episode
    """
    wins = 0
    ties = 0
    losses = 0
    total_score = 0
    episode_results = []
    episode_scores = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            if SB3_AVAILABLE and model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Fallback: simple threshold policy
                current_score = info.get("current_score", 0)
                action = 1 if current_score > 50 else 0
            
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Get final scores
        if terminated or truncated:
            if "final_scores" in info:
                final_scores = info["final_scores"]
            elif "player_scores" in info:
                final_scores = info["player_scores"]
            else:
                final_scores = [0, 0]
            
            agent_score = final_scores[0]
            opponent_scores = final_scores[1:] if len(final_scores) > 1 else [0]
            max_opponent_score = max(opponent_scores) if opponent_scores else 0
            total_score += agent_score
            episode_scores.append(agent_score)
            
            if agent_score > max_opponent_score:
                wins += 1
                episode_results.append(1)
            elif agent_score == max_opponent_score:
                ties += 1
                episode_results.append(0)
            else:
                losses += 1
                episode_results.append(-1)
    
    win_rate = wins / n_episodes
    avg_score = total_score / n_episodes
    
    return win_rate, avg_score, episode_results, episode_scores


def calculate_expected_wins(win_rate: float, n_matches: int) -> Tuple[float, float, float]:
    """
    Calculate expected wins, with confidence intervals.
    
    Returns:
        expected_wins: Expected number of wins
        lower_bound: Lower bound (95% confidence)
        upper_bound: Upper bound (95% confidence)
    """
    expected_wins = win_rate * n_matches
    # Binomial confidence interval (normal approximation)
    std = np.sqrt(win_rate * (1 - win_rate) * n_matches)
    lower_bound = max(0, expected_wins - 1.96 * std)
    upper_bound = min(n_matches, expected_wins + 1.96 * std)
    
    return expected_wins, lower_bound, upper_bound


class ProgressCallback:
    """Callback to track training progress."""
    def __init__(self, env: BankEnv, eval_episodes: int = 20):
        self.env = env
        self.eval_episodes = eval_episodes
        self.timesteps = []
        self.win_rates = []
        self.avg_scores = []
    
    def __call__(self, locals_, globals_):
        """Called during training to track progress."""
        if SB3_AVAILABLE:
            self.num_timesteps = locals_['self'].num_timesteps
            if self.num_timesteps % TRACK_PROGRESS_INTERVAL == 0:
                win_rate, avg_score, _, _ = evaluate_agent(self.env, locals_['self'], self.eval_episodes)
                self.timesteps.append(self.num_timesteps)
                self.win_rates.append(win_rate)
                self.avg_scores.append(avg_score)
                print(f"Progress at {self.num_timesteps} timesteps: Win rate = {win_rate:.3f}, Avg score = {avg_score:.2f}")


# ============================================================================
# TRAINING
# ============================================================================

def train_agent():
    """Train the RL agent."""
    print("=" * 80)
    print("Starting RL Training")
    print("=" * 80)
    print(f"Environment rounds: {ENV_ROUNDS}")
    print(f"Opponents: {len(ENV_OPPONENTS)} opponent(s)")
    print(f"Training timesteps: {TRAIN_TOTAL_TIMESTEPS}")
    print(f"Algorithm: {TRAIN_ALGORITHM}")
    print()
    
    # Create environment
    env = BankEnv(
        rounds=ENV_ROUNDS,
        opponents=ENV_OPPONENTS,
        max_round_length=ENV_MAX_ROUND_LENGTH,
        verbose=False,
    )
    
    # Wrap environment for monitoring
    if SB3_AVAILABLE:
        env = Monitor(env, filename=None, allow_early_resets=True)
    
    # Train the agent
    model = None
    training_win_rates = []
    training_timesteps = []
    training_avg_scores = []
    
    if SB3_AVAILABLE:
        print(f"Training with stable-baselines3 using {TRAIN_ALGORITHM} algorithm...")
        # Define a better network architecture
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        import torch as th
        import torch.nn as nn
        
        class CustomMLP(BaseFeaturesExtractor):
            """Custom MLP with better architecture for the bank game."""
            def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 128):
                super().__init__(observation_space, features_dim)
                n_input = observation_space.shape[0]
                self.net = nn.Sequential(
                    nn.Linear(n_input, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, features_dim),
                    nn.ReLU(),
                )
            
            def forward(self, observations: th.Tensor) -> th.Tensor:
                return self.net(observations)
        
        # Common policy kwargs for policy-based algorithms (PPO, A2C)
        policy_kwargs = dict(
            features_extractor_class=CustomMLP,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=[256, 256],  # Additional layers in the policy/value networks
        )
        
        # Create model based on selected algorithm
        config = ALGORITHM_CONFIG.copy()
        
        if TRAIN_ALGORITHM == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config["learning_rate"],
                n_steps=config["n_steps"],
                batch_size=config["batch_size"],
                n_epochs=config["n_epochs"],
                gamma=config["gamma"],
                gae_lambda=config["gae_lambda"],
                clip_range=config["clip_range"],
                ent_coef=config["ent_coef"],
                vf_coef=config["vf_coef"],
                max_grad_norm=config["max_grad_norm"],
                policy_kwargs=policy_kwargs,
                verbose=TRAIN_VERBOSE,
                tensorboard_log="./tb_logs/",
            )
        elif TRAIN_ALGORITHM == "A2C":
            model = A2C(
                "MlpPolicy",
                env,
                learning_rate=config["learning_rate"],
                n_steps=config["n_steps"],
                gamma=config["gamma"],
                gae_lambda=config["gae_lambda"],
                ent_coef=config["ent_coef"],
                vf_coef=config["vf_coef"],
                max_grad_norm=config["max_grad_norm"],
                use_rms_prop=config["use_rms_prop"],
                policy_kwargs=policy_kwargs,
                verbose=TRAIN_VERBOSE,
                tensorboard_log="./tb_logs/",
            )
        elif TRAIN_ALGORITHM == "DQN":
            # DQN uses different policy kwargs (no features extractor needed)
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=config["learning_rate"],
                buffer_size=config["buffer_size"],
                learning_starts=config["learning_starts"],
                batch_size=config["batch_size"],
                tau=config["tau"],
                gamma=config["gamma"],
                train_freq=config["train_freq"],
                gradient_steps=config["gradient_steps"],
                target_update_interval=config["target_update_interval"],
                exploration_fraction=config["exploration_fraction"],
                exploration_initial_eps=config["exploration_initial_eps"],
                exploration_final_eps=config["exploration_final_eps"],
                policy_kwargs=dict(net_arch=[256, 256]),
                verbose=TRAIN_VERBOSE,
                tensorboard_log="./tb_logs/",
            )
        else:
            raise ValueError(f"Unknown algorithm: {TRAIN_ALGORITHM}. Choose from: {list(ALGORITHM_CONFIGS.keys())}")
        
        # Set up progress tracking
        progress_callback = ProgressCallback(env, PROGRESS_EVAL_EPISODES)
        
        # Custom callback to track progress
        from stable_baselines3.common.callbacks import BaseCallback
        
        class ProgressTrackingCallback(BaseCallback):
            def __init__(self, progress_callback, eval_env, verbose=0):
                super().__init__(verbose)
                self.progress_callback = progress_callback
                self.eval_env = eval_env
                self.last_eval_step = 0
            
            def _on_step(self) -> bool:
                # Evaluate periodically during training
                if self.num_timesteps - self.last_eval_step >= TRACK_PROGRESS_INTERVAL:
                    self.last_eval_step = self.num_timesteps
                    try:
                        win_rate, avg_score, _, _ = evaluate_agent(
                            self.eval_env, self.model, PROGRESS_EVAL_EPISODES
                        )
                        self.progress_callback.timesteps.append(self.num_timesteps)
                        self.progress_callback.win_rates.append(win_rate)
                        self.progress_callback.avg_scores.append(avg_score)
                        print(f"Progress at {self.num_timesteps} timesteps: Win rate = {win_rate:.3f}, Avg score = {avg_score:.2f}")
                    except Exception as e:
                        print(f"Warning: Could not evaluate progress at {self.num_timesteps}: {e}")
                return True
        
        # Create separate evaluation environment for progress tracking
        eval_env = BankEnv(
            rounds=ENV_ROUNDS,
            opponents=ENV_OPPONENTS,
            max_round_length=ENV_MAX_ROUND_LENGTH,
            verbose=False,
        )
        callback = ProgressTrackingCallback(progress_callback, eval_env)
        
        # Train
        model.learn(
            total_timesteps=TRAIN_TOTAL_TIMESTEPS,
            callback=callback,
            progress_bar=True,
        )
        
        training_win_rates = progress_callback.win_rates
        training_timesteps = progress_callback.timesteps
        training_avg_scores = progress_callback.avg_scores
        
        # Close evaluation environment
        eval_env.close()
        
        # Save model
        if SAVE_MODEL:
            model.save(MODEL_SAVE_PATH)
            print(f"\nModel saved to {MODEL_SAVE_PATH}")
    else:
        print("Warning: stable-baselines3 not available. Using placeholder.")
        print("Install with: pip install stable-baselines3")
        print("Continuing with placeholder training data...")
        # Create dummy progress data for visualization
        training_timesteps = [i * TRACK_PROGRESS_INTERVAL for i in range(1, TRAIN_TOTAL_TIMESTEPS // TRACK_PROGRESS_INTERVAL + 1)]
        training_win_rates = [0.3 + 0.2 * (i / len(training_timesteps)) for i in range(len(training_timesteps))]  # Dummy data
        training_avg_scores = [50 + 30 * (i / len(training_timesteps)) for i in range(len(training_timesteps))]  # Dummy data
    
    env.close()
    
    return model, training_win_rates, training_timesteps, training_avg_scores


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_trained_agent(model):
    """Evaluate the trained agent."""
    print("\n" + "=" * 80)
    print("Evaluating Trained Agent")
    print("=" * 80)
    
    if model is None and not SB3_AVAILABLE:
        print("Warning: No trained model available. Using simple threshold policy for evaluation.")
    
    # Create evaluation environment
    eval_env = BankEnv(
        rounds=ENV_ROUNDS,
        opponents=ENV_OPPONENTS,
        max_round_length=ENV_MAX_ROUND_LENGTH,
        verbose=False,
    )
    
    # Evaluate
    win_rate, avg_score, episode_results, episode_scores = evaluate_agent(
        eval_env, model, EVAL_N_EPISODES
    )
    
    print(f"Evaluation episodes: {EVAL_N_EPISODES}")
    print(f"Win rate: {win_rate:.3f} ({win_rate*100:.1f}%)")
    print(f"Average score: {avg_score:.2f}")
    print(f"Wins: {sum(1 for r in episode_results if r == 1)}")
    print(f"Ties: {sum(1 for r in episode_results if r == 0)}")
    print(f"Losses: {sum(1 for r in episode_results if r == -1)}")
    
    eval_env.close()
    
    return win_rate, avg_score, episode_results, episode_scores


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comprehensive_plot(
    training_win_rates: List[float],
    training_timesteps: List[int],
    training_avg_scores: List[float],
    post_train_win_rate: float,
    post_train_avg_score: float,
    episode_scores: List[float],
    expected_wins_data: List[Tuple[int, float, float, float]],
):
    """Create a comprehensive plot with all statistics."""
    fig, axes = plt.subplots(2, 2, figsize=PLOT_FIGURE_SIZE)
    fig.suptitle('RL Training Statistics', fontsize=16, fontweight='bold')
    
    # 1. Training Progression: Win Rate
    ax1 = axes[0, 0]
    if training_timesteps and training_win_rates:
        ax1.plot(training_timesteps, training_win_rates, 'b-', linewidth=2, label='Win Rate')
        ax1.axhline(y=post_train_win_rate, color='r', linestyle='--', linewidth=1.5, label=f'Final Win Rate ({post_train_win_rate:.3f})')
        ax1.set_xlabel('Training Timesteps')
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Training Progression: Win Rate')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0, 1])
    else:
        ax1.text(0.5, 0.5, 'No training progress data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Training Progression: Win Rate')
    
    # 2. Training Progression: Average Score
    ax2 = axes[0, 1]
    if training_timesteps and training_avg_scores:
        ax2.plot(training_timesteps, training_avg_scores, 'g-', linewidth=2, label='Avg Score')
        ax2.axhline(y=post_train_avg_score, color='r', linestyle='--', linewidth=1.5, label=f'Final Avg Score ({post_train_avg_score:.1f})')
        ax2.set_xlabel('Training Timesteps')
        ax2.set_ylabel('Average Score')
        ax2.set_title('Training Progression: Average Score')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No training progress data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Training Progression: Average Score')
    
    # 3. Post-Training Win Rate and Expected Wins
    ax3 = axes[1, 0]
    if expected_wins_data:
        matches = [x[0] for x in expected_wins_data]
        expected = [x[1] for x in expected_wins_data]
        lower = [x[2] for x in expected_wins_data]
        upper = [x[3] for x in expected_wins_data]
        
        ax3.plot(matches, expected, 'o-', linewidth=2, markersize=8, label='Expected Wins', color='purple')
        ax3.fill_between(matches, lower, upper, alpha=0.3, color='purple', label='95% Confidence Interval')
        ax3.axhline(y=post_train_win_rate * max(matches) if matches else 0, color='r', linestyle='--', linewidth=1, label=f'Win Rate Line ({post_train_win_rate:.3f})')
        ax3.set_xlabel('Number of Matches')
        ax3.set_ylabel('Expected Wins')
        ax3.set_title('Expected Wins in X Matches')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_xscale('log')
    else:
        ax3.text(0.5, 0.5, 'No expected wins data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Expected Wins in X Matches')
    
    # 4. Score Distribution
    ax4 = axes[1, 1]
    if episode_scores:
        ax4.hist(episode_scores, bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax4.axvline(x=post_train_avg_score, color='r', linestyle='--', linewidth=2, label=f'Mean ({post_train_avg_score:.1f})')
        ax4.set_xlabel('Final Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Post-Training Score Distribution')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No score data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Post-Training Score Distribution')
    
    plt.tight_layout()
    
    if SAVE_PLOT:
        plt.savefig(PLOT_SAVE_PATH, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"\nPlot saved to {PLOT_SAVE_PATH}")
    
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run training and evaluation."""
    # Train the agent
    model, training_win_rates, training_timesteps, training_avg_scores = train_agent()
    
    # Evaluate the trained agent
    post_train_win_rate, post_train_avg_score, episode_results, episode_scores = evaluate_trained_agent(model)
    
    # Calculate expected wins for different match counts
    print("\n" + "=" * 80)
    print("Expected Wins Projection")
    print("=" * 80)
    expected_wins_data = []
    for n_matches in EVAL_N_MATCHES_FOR_PROJECTION:
        expected, lower, upper = calculate_expected_wins(post_train_win_rate, n_matches)
        expected_wins_data.append((n_matches, expected, lower, upper))
        print(f"{n_matches} matches: {expected:.1f} wins (95% CI: {lower:.1f} - {upper:.1f})")
    
    # Create comprehensive visualization
    print("\n" + "=" * 80)
    print("Generating Visualization")
    print("=" * 80)
    create_comprehensive_plot(
        training_win_rates,
        training_timesteps,
        training_avg_scores,
        post_train_win_rate,
        post_train_avg_score,
        episode_scores,
        expected_wins_data,
    )
    
    print("\n" + "=" * 80)
    print("Training and Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

