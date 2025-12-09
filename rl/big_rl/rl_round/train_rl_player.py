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
from typing import List, Tuple, Dict, Optional
import gymnasium as gym

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    raise ImportError("stable-baselines3 not available. Please install it with: pip install stable-baselines3")

try:
    from bank_gym import BankEnv
    from players import ThersholdPlayer, GreedyPlayer, GreedyPlayerK, SesquaGreedyPlayer, ProbabilisticPlayer, TrollPlayer
except ModuleNotFoundError as e:
    if 'game_theory' in str(e):
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        from bank_gym import BankEnv
        from players import ThersholdPlayer, GreedyPlayer, GreedyPlayerK, SesquaGreedyPlayer, ProbabilisticPlayer, TrollPlayer
    else:
        raise

ENV_ROUNDS = 10
ENV_OPPONENTS = [
    ThersholdPlayer(threshold=100),
    ThersholdPlayer(threshold=180),
    GreedyPlayer(),
    SesquaGreedyPlayer(),
    ProbabilisticPlayer(probability=0.2),
    TrollPlayer(),
]
ENV_MAX_ROUND_LENGTH = 1000

PPO_CONFIG = {
    "learning_rate": 3e-4,   # faster learning
    "n_steps": 1024,         # slightly shorter rollouts
    "batch_size": 256,       # bigger batch, less noisy
    "n_epochs": 10,          # OK
    "gamma": 0.995,          # longer horizon since reward is delayed
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.02,        # encourage exploration more
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

TRAIN_TOTAL_TIMESTEPS = 1000000
TRAIN_VERBOSE = 1
TRAIN_LOG_INTERVAL = 10

EVAL_N_EPISODES = 1000
TRACK_PROGRESS_INTERVAL = 1000
PROGRESS_EVAL_EPISODES = 20

SAVE_MODEL = True
MODEL_SAVE_PATH = str(script_dir / "RL_data" / "rl_bank_model_ppo_troll.zip")

PLOT_FIGURE_SIZE = (14, 10)
PLOT_DPI = 100
SAVE_PLOT = True
PLOT_SAVE_PATH = str(script_dir / "RL_data" / "rl_training_stats_ppo_troll.png")


def evaluate_agent(env, model, n_episodes: int = 100) -> Tuple[float, float, List[int], List[float]]:
    """Evaluate the agent and return statistics."""
    wins = 0
    ties = 0
    losses = 0
    total_score = 0
    episode_results = []
    episode_scores = []
    
    # Check if env is a VecEnv (has step method that returns 4 values)
    is_vec_env = hasattr(env, 'step') and not isinstance(env, BankEnv)
    # Get underlying env for fallback logic if VecEnv
    underlying_env = env.venv.envs[0] if is_vec_env and hasattr(env, 'venv') else env
    
    for episode in range(n_episodes):
        if is_vec_env:
            obs = env.reset()
            # VecEnv returns batched obs, get first element
            if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
                obs = obs[0]
        else:
            obs, info = env.reset()
            if isinstance(info, dict):
                info = [info]
        
        terminated = False
        truncated = False
        info = [{}] if is_vec_env else (info if isinstance(info, list) else [info])
        
        while not (terminated or truncated):
            if SB3_AVAILABLE and model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                current_info = info[0] if isinstance(info, list) else info
                current_score = current_info.get("current_score", 0)
                action = 1 if current_score > 50 else 0
            
            if is_vec_env:
                # VecEnv expects batched actions
                action_batch = np.array([action])
                obs, rewards, dones, infos = env.step(action_batch)
                terminated = bool(dones[0])
                truncated = False  # VecEnv doesn't distinguish terminated/truncated
                # VecEnv returns batched obs, get first element
                if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
                    obs = obs[0]
                info = infos if isinstance(infos, list) else [infos]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                if isinstance(info, dict):
                    info = [info]
        
        # Get final scores
        if terminated or truncated:
            current_info = info[0] if isinstance(info, list) else info
            if "final_scores" in current_info:
                final_scores = current_info["final_scores"]
            elif "player_scores" in current_info:
                final_scores = current_info["player_scores"]
            else:
                # Fallback: try to get from environment if available
                if hasattr(underlying_env, '_final_scores') and underlying_env._final_scores is not None:
                    final_scores = underlying_env._final_scores.tolist() if hasattr(underlying_env._final_scores, 'tolist') else list(underlying_env._final_scores)
                elif hasattr(underlying_env, '_player_scores') and underlying_env._player_scores is not None:
                    final_scores = underlying_env._player_scores.tolist() if hasattr(underlying_env._player_scores, 'tolist') else list(underlying_env._player_scores)
                else:
                    n_opponents = underlying_env.n_opponents if hasattr(underlying_env, 'n_opponents') else 1
                    final_scores = [0.0] * (1 + n_opponents)
            
            # Ensure we have at least one score (the agent)
            if len(final_scores) == 0:
                final_scores = [0.0]
            
            agent_score = float(final_scores[0])
            opponent_scores = [float(s) for s in final_scores[1:]] if len(final_scores) > 1 else []
            max_opponent_score = max(opponent_scores) if opponent_scores else 0.0
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
    
    if SB3_AVAILABLE:
        def make_env():
            return BankEnv(
                rounds=ENV_ROUNDS,
                opponents=ENV_OPPONENTS,
                max_round_length=ENV_MAX_ROUND_LENGTH,
                verbose=False,
            )
        
        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0)
        env = vec_env
    else:
        env = BankEnv(
            rounds=ENV_ROUNDS,
            opponents=ENV_OPPONENTS,
            max_round_length=ENV_MAX_ROUND_LENGTH,
            verbose=False,
        )
    
    model = None
    training_win_rates = []
    training_timesteps = []
    training_avg_scores = []
    
    if SB3_AVAILABLE:
        print("Training with stable-baselines3 using PPO...")
        
        policy_kwargs = dict(
            net_arch=[128, 128],  # let PPO build a simple MLP
        )
        
        # Ensure tensorboard log directory exists
        tb_log_dir = script_dir / "RL_data" / "tb_logs"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            **PPO_CONFIG,
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
        
        def make_eval_env():
            return BankEnv(
                rounds=ENV_ROUNDS,
                opponents=ENV_OPPONENTS,
                max_round_length=ENV_MAX_ROUND_LENGTH,
                verbose=False,
            )
        
        eval_vec_env = DummyVecEnv([make_eval_env])
        eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0)
        # Sync normalization stats from training env
        eval_vec_env.obs_rms = vec_env.obs_rms
        eval_vec_env.ret_rms = vec_env.ret_rms
        eval_env = eval_vec_env
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
    
    if SB3_AVAILABLE:
        def make_eval_env():
            return BankEnv(
                rounds=ENV_ROUNDS,
                opponents=ENV_OPPONENTS,
                max_round_length=ENV_MAX_ROUND_LENGTH,
                verbose=False,
            )
        
        eval_vec_env = DummyVecEnv([make_eval_env])
        eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0)
        # If model exists and has normalization stats, sync them
        if model is not None and hasattr(model, 'get_vec_normalize_env') and model.get_vec_normalize_env() is not None:
            train_vec_env = model.get_vec_normalize_env()
            eval_vec_env.obs_rms = train_vec_env.obs_rms
            eval_vec_env.ret_rms = train_vec_env.ret_rms
        eval_env = eval_vec_env
    else:
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


def evaluate_against_opponents(model, n_episodes: int = 1000) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate the trained agent against multiple opponent types in 1v1 matches.
    
    Returns:
        Tuple of (win_rates, tie_rates) dictionaries
    """
    print("\n" + "=" * 80)
    print("Evaluating Against Multiple Opponents (1v1)")
    print("=" * 80)
    
    opponents = [
        ("Threshold 100", ThersholdPlayer(threshold=100)),
        ("Threshold 200", ThersholdPlayer(threshold=200)),
        ("Greedy", GreedyPlayer()),
        ("SesquaGreedy", SesquaGreedyPlayer()),
        ("Probabilistic 0.2", ProbabilisticPlayer(probability=0.2)),
    ]
    
    win_rates = {}
    tie_rates = {}
    
    for opponent_name, opponent in opponents:
        print(f"\nEvaluating against {opponent_name}...")
        
        if SB3_AVAILABLE:
            def make_eval_env():
                return BankEnv(
                    rounds=ENV_ROUNDS,
                    opponents=[opponent],
                    max_round_length=ENV_MAX_ROUND_LENGTH,
                    verbose=False,
                )
            
            eval_vec_env = DummyVecEnv([make_eval_env])
            eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0)
            # If model exists and has normalization stats, sync them
            if model is not None and hasattr(model, 'get_vec_normalize_env') and model.get_vec_normalize_env() is not None:
                train_vec_env = model.get_vec_normalize_env()
                eval_vec_env.obs_rms = train_vec_env.obs_rms
                eval_vec_env.ret_rms = train_vec_env.ret_rms
            eval_env = eval_vec_env
        else:
            eval_env = BankEnv(
                rounds=ENV_ROUNDS,
                opponents=[opponent],
                max_round_length=ENV_MAX_ROUND_LENGTH,
                verbose=False,
            )
        
        win_rate, avg_score, episode_results, episode_scores = evaluate_agent(
            eval_env, model, n_episodes
        )
        
        # Calculate tie rate
        ties = sum(1 for r in episode_results if r == 0)
        tie_rate = ties / n_episodes
        
        win_rates[opponent_name] = win_rate
        tie_rates[opponent_name] = tie_rate
        print(f"  Win rate: {win_rate:.3f} ({win_rate*100:.1f}%)")
        print(f"  Tie rate: {tie_rate:.3f} ({tie_rate*100:.1f}%)")
        print(f"  Non-loss rate: {win_rate + tie_rate:.3f} ({(win_rate + tie_rate)*100:.1f}%)")
        print(f"  Average score: {avg_score:.2f}")
        print(f"  Wins: {sum(1 for r in episode_results if r == 1)}, "
              f"Ties: {ties}, "
              f"Losses: {sum(1 for r in episode_results if r == -1)}")
        
        eval_env.close()
    
    return win_rates, tie_rates


def create_training_plot(
    training_win_rates: List[float],
    training_timesteps: List[int],
    training_avg_scores: List[float],
    post_train_win_rate: float,
    post_train_avg_score: float,
    opponent_win_rates: Optional[Dict[str, float]] = None,
    opponent_tie_rates: Optional[Dict[str, float]] = None,
):
    """Create a plot showing win rate and score over training, plus opponent win rates."""
    if opponent_win_rates:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RL Training Statistics', fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        ax2 = axes[0, 1]
        ax3 = axes[1, 0]
        ax4 = axes[1, 1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=PLOT_FIGURE_SIZE)
        fig.suptitle('RL Training Statistics', fontsize=16, fontweight='bold')
        ax1 = axes[0]
        ax2 = axes[1]
    
    # Training progression: Win Rate
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
    
    # Training progression: Average Score
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
    
    # Opponent win rates and tie rates stacked bar chart
    if opponent_win_rates and opponent_tie_rates:
        opponent_names = list(opponent_win_rates.keys())
        win_rate_values = list(opponent_win_rates.values())
        tie_rate_values = list(opponent_tie_rates.values())
        
        # Create stacked bar chart: win rates at bottom, tie rates on top
        bars1 = ax3.bar(opponent_names, win_rate_values, color='green', alpha=0.7, 
                        edgecolor='black', label='Win Rate')
        bars2 = ax3.bar(opponent_names, tie_rate_values, bottom=win_rate_values, 
                        color='blue', alpha=0.7, edgecolor='black', label='Tie Rate')
        
        ax3.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='50% Reference')
        ax3.set_ylabel('Rate')
        ax3.set_title('Win Rate and Tie Rate vs Different Opponents (1v1, 1000 games each)')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, (bar1, bar2, win_val, tie_val) in enumerate(zip(bars1, bars2, win_rate_values, tie_rate_values)):
            # Label for win rate (at middle of win bar)
            win_height = bar1.get_height()
            if win_height > 0.05:  # Only label if bar is tall enough
                ax3.text(bar1.get_x() + bar1.get_width()/2., win_height/2,
                        f'W:{win_val:.3f}',
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            
            # Label for tie rate (at middle of tie bar)
            tie_height = bar2.get_height()
            total_height = win_height + tie_height
            if tie_height > 0.05:  # Only label if bar is tall enough
                ax3.text(bar2.get_x() + bar2.get_width()/2., win_height + tie_height/2,
                        f'T:{tie_val:.3f}',
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            
            # Label for total non-loss rate at top
            if total_height > 0.1:
                ax3.text(bar1.get_x() + bar1.get_width()/2., total_height + 0.02,
                        f'{(win_val + tie_val):.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Hide the 4th subplot if we have opponent data
        ax4.axis('off')
    
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
    
    # Evaluate against multiple opponents
    opponent_win_rates, opponent_tie_rates = evaluate_against_opponents(model, n_episodes=1000)
    
    print("\n" + "=" * 80)
    print("Generating Visualization")
    print("=" * 80)
    create_training_plot(
        training_win_rates,
        training_timesteps,
        training_avg_scores,
        post_train_win_rate,
        post_train_avg_score,
        opponent_win_rates,
        opponent_tie_rates,
    )
    
    print("\n" + "=" * 80)
    print("Training and Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

