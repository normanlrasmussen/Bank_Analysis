"""RL Training Environment for Bank Game."""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

parent_dir = str(project_root.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import gymnasium as gym

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    raise ImportError("stable-baselines3 not available. Please install it with: pip install stable-baselines3")

try:
    from bank_gym import BankEnv
    from players import ThersholdPlayer, GreedyPlayer, GreedyPlayerK, SesquaGreedyPlayer, ProbabilisticPlayer
except ModuleNotFoundError as e:
    if 'game_theory' in str(e):
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        from bank_gym import BankEnv
        from players import ThersholdPlayer, GreedyPlayer, GreedyPlayerK, SesquaGreedyPlayer, ProbabilisticPlayer
    else:
        raise

ENV_ROUNDS = 10
ENV_OPPONENTS = [
    ThersholdPlayer(threshold=100),
]
ENV_MAX_ROUND_LENGTH = 100

PPO_CONFIG = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

TRAIN_TOTAL_TIMESTEPS = 500000
TRAIN_VERBOSE = 1
TRAIN_LOG_INTERVAL = 10

EVAL_N_EPISODES = 1000
EVAL_N_MATCHES_FOR_PROJECTION = [10, 50, 100, 500, 1000]

TRACK_PROGRESS_INTERVAL = 1000
PROGRESS_EVAL_EPISODES = 20

SAVE_MODEL = True
MODEL_SAVE_PATH = str(script_dir / "RL_data" / "rl_bank_model_ppo_100.zip")

PLOT_FIGURE_SIZE = (14, 10)
PLOT_DPI = 100
SAVE_PLOT = True
PLOT_SAVE_PATH = str(script_dir / "RL_data" / "rl_training_stats_ppo_100.png")


def evaluate_agent(env: BankEnv, model, n_episodes: int = 100) -> Tuple[float, float, List[int], List[float]]:
    """Evaluate the agent and return statistics."""
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
    """Calculate expected wins with 95% confidence intervals."""
    expected_wins = win_rate * n_matches
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
        """Track progress during training."""
        if SB3_AVAILABLE:
            self.num_timesteps = locals_['self'].num_timesteps
            if self.num_timesteps % TRACK_PROGRESS_INTERVAL == 0:
                win_rate, avg_score, _, _ = evaluate_agent(self.env, locals_['self'], self.eval_episodes)
                self.timesteps.append(self.num_timesteps)
                self.win_rates.append(win_rate)
                self.avg_scores.append(avg_score)
                print(f"Progress at {self.num_timesteps} timesteps: Win rate = {win_rate:.3f}, Avg score = {avg_score:.2f}")


def train_agent():
    """Train the RL agent."""
    print("=" * 80)
    print("Starting PPO Training")
    print("=" * 80)
    print(f"Environment rounds: {ENV_ROUNDS}")
    print(f"Opponents: {len(ENV_OPPONENTS)} opponent(s)")
    print(f"Training timesteps: {TRAIN_TOTAL_TIMESTEPS}")
    print()
    
    env = BankEnv(
        rounds=ENV_ROUNDS,
        opponents=ENV_OPPONENTS,
        max_round_length=ENV_MAX_ROUND_LENGTH,
        verbose=False,
    )
    
    if SB3_AVAILABLE:
        env = Monitor(env, filename=None, allow_early_resets=True)
    
    model = None
    training_win_rates = []
    training_timesteps = []
    training_avg_scores = []
    
    if SB3_AVAILABLE:
        print("Training with stable-baselines3 using PPO...")
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        import torch as th
        import torch.nn as nn
        
        class CustomMLP(BaseFeaturesExtractor):
            """Custom MLP for the bank game."""
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
        
        policy_kwargs = dict(
            features_extractor_class=CustomMLP,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=[256, 256],
        )
        
        # Ensure tensorboard log directory exists
        tb_log_dir = script_dir / "RL_data" / "tb_logs"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=PPO_CONFIG["learning_rate"],
            n_steps=PPO_CONFIG["n_steps"],
            batch_size=PPO_CONFIG["batch_size"],
            n_epochs=PPO_CONFIG["n_epochs"],
            gamma=PPO_CONFIG["gamma"],
            gae_lambda=PPO_CONFIG["gae_lambda"],
            clip_range=PPO_CONFIG["clip_range"],
            ent_coef=PPO_CONFIG["ent_coef"],
            vf_coef=PPO_CONFIG["vf_coef"],
            max_grad_norm=PPO_CONFIG["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            verbose=TRAIN_VERBOSE,
            tensorboard_log=str(tb_log_dir),
        )
        
        progress_callback = ProgressCallback(env, PROGRESS_EVAL_EPISODES)
        
        from stable_baselines3.common.callbacks import BaseCallback
        
        class ProgressTrackingCallback(BaseCallback):
            def __init__(self, progress_callback, eval_env, verbose=0):
                super().__init__(verbose)
                self.progress_callback = progress_callback
                self.eval_env = eval_env
                self.last_eval_step = 0
            
            def _on_step(self) -> bool:
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
        
        eval_env = BankEnv(
            rounds=ENV_ROUNDS,
            opponents=ENV_OPPONENTS,
            max_round_length=ENV_MAX_ROUND_LENGTH,
            verbose=False,
        )
        callback = ProgressTrackingCallback(progress_callback, eval_env)
        
        model.learn(
            total_timesteps=TRAIN_TOTAL_TIMESTEPS,
            callback=callback,
            progress_bar=True,
        )
        
        training_win_rates = progress_callback.win_rates
        training_timesteps = progress_callback.timesteps
        training_avg_scores = progress_callback.avg_scores
        
        eval_env.close()
        
        if SAVE_MODEL:
            Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
            model.save(MODEL_SAVE_PATH)
            print(f"\nModel saved to {MODEL_SAVE_PATH}")
    else:
        print("Warning: stable-baselines3 not available. Using placeholder.")
        print("Install with: pip install stable-baselines3")
        print("Continuing with placeholder training data...")
        training_timesteps = [i * TRACK_PROGRESS_INTERVAL for i in range(1, TRAIN_TOTAL_TIMESTEPS // TRACK_PROGRESS_INTERVAL + 1)]
        training_win_rates = [0.3 + 0.2 * (i / len(training_timesteps)) for i in range(len(training_timesteps))]
        training_avg_scores = [50 + 30 * (i / len(training_timesteps)) for i in range(len(training_timesteps))]
    
    env.close()
    
    return model, training_win_rates, training_timesteps, training_avg_scores


def evaluate_trained_agent(model):
    """Evaluate the trained agent."""
    print("\n" + "=" * 80)
    print("Evaluating Trained Agent")
    print("=" * 80)
    
    if model is None and not SB3_AVAILABLE:
        print("Warning: No trained model available. Using simple threshold policy for evaluation.")
    
    eval_env = BankEnv(
        rounds=ENV_ROUNDS,
        opponents=ENV_OPPONENTS,
        max_round_length=ENV_MAX_ROUND_LENGTH,
        verbose=False,
    )
    
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
        Path(PLOT_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOT_SAVE_PATH, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"\nPlot saved to {PLOT_SAVE_PATH}")
    
    plt.show()


def main():
    """Main function to run training and evaluation."""
    model, training_win_rates, training_timesteps, training_avg_scores = train_agent()
    
    post_train_win_rate, post_train_avg_score, episode_results, episode_scores = evaluate_trained_agent(model)
    
    print("\n" + "=" * 80)
    print("Expected Wins Projection")
    print("=" * 80)
    expected_wins_data = []
    for n_matches in EVAL_N_MATCHES_FOR_PROJECTION:
        expected, lower, upper = calculate_expected_wins(post_train_win_rate, n_matches)
        expected_wins_data.append((n_matches, expected, lower, upper))
        print(f"{n_matches} matches: {expected:.1f} wins (95% CI: {lower:.1f} - {upper:.1f})")
    
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

