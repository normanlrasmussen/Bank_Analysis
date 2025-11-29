"""RL Training Environment for Bank Game using SAC-Discrete with CleanRL."""

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
from typing import List, Tuple, Optional
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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
ENV_MAX_ROUND_LENGTH = 1000

SAC_DISCRETE_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 100000,
    "learning_starts": 5000,  # Reduced from 1000 to start learning earlier (but still need buffer)
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "alpha": 0.2,  # Temperature parameter for entropy (can be tuned)
    "target_update_interval": 1,
    "gradient_steps": 4,  # Increased from 1 to 4 for more learning per update (PPO does 10 epochs)
    "train_freq": 4,  # Train every 4 steps instead of every step (more efficient)
}

TRAIN_TOTAL_TIMESTEPS = 500000  # Match PPO's training timesteps for fair comparison
TRAIN_VERBOSE = 1
TRAIN_LOG_INTERVAL = 10

EVAL_N_EPISODES = 1000
EVAL_N_MATCHES_FOR_PROJECTION = [10, 50, 100, 500, 1000]

TRACK_PROGRESS_INTERVAL = 1000
PROGRESS_EVAL_EPISODES = 20

SAVE_MODEL = True
MODEL_SAVE_PATH = str(script_dir / "RL_data" / "rl_bank_model_sac_discrete.zip")

PLOT_FIGURE_SIZE = (14, 10)
PLOT_DPI = 100
SAVE_PLOT = True
PLOT_SAVE_PATH = str(script_dir / "RL_data" / "rl_training_stats_sac_discrete.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Q-Network for SAC-Discrete."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Actor(nn.Module):
    """Actor network for SAC-Discrete."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.net(obs)
        return F.softmax(logits, dim=-1)
    
    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """Get raw logits for efficiency (avoids double forward pass)."""
        return self.net(obs)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        probs = self.forward(obs)
        if deterministic:
            action = torch.argmax(probs, dim=-1).item()
        else:
            dist = Categorical(probs)
            action = dist.sample().item()
        return action, probs


class ReplayBuffer:
    """Optimized replay buffer using numpy arrays for faster sampling."""
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate numpy arrays for efficiency
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, obs, action, reward, next_obs, done):
        """Store transition in buffer."""
        self.obs[self.ptr] = np.asarray(obs, dtype=np.float32)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = np.asarray(next_obs, dtype=np.float32)
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """Sample batch of transitions."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )
    
    def __len__(self):
        return self.size


class SACDiscreteAgent:
    """SAC-Discrete agent implementation based on CleanRL."""
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        buffer_size: int = 100000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: float = 0.2,
        learning_starts: int = 1000,
        target_update_interval: int = 1,
        gradient_steps: int = 1,
        train_freq: int = 1,
        device: torch.device = DEVICE,
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.learning_starts = learning_starts
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.train_freq = train_freq
        self.device = device
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        # Networks
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.q1 = QNetwork(obs_dim, action_dim).to(device)
        self.q2 = QNetwork(obs_dim, action_dim).to(device)
        self.q1_target = QNetwork(obs_dim, action_dim).to(device)
        self.q2_target = QNetwork(obs_dim, action_dim).to(device)
        
        # Copy weights to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=learning_rate)
        
        # Replay buffer (optimized with pre-allocated arrays)
        self.replay_buffer = ReplayBuffer(buffer_size, obs_dim)
        
        # Training state
        self.num_timesteps = 0
        self.num_updates = 0
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Select an action given an observation (optimized with tensor caching)."""
        # Use from_numpy for zero-copy when possible, ensure contiguous
        obs_array = np.ascontiguousarray(obs, dtype=np.float32)
        obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.no_grad():
            action, _ = self.actor.get_action(obs_tensor, deterministic=deterministic)
        return action
    
    def update(self) -> dict:
        """Optimized update function with reduced tensor operations."""
        if len(self.replay_buffer) < self.learning_starts:
            return {}
        
        losses = {"q1_loss": 0.0, "q2_loss": 0.0, "actor_loss": 0.0}
        
        for _ in range(self.gradient_steps):
            # Sample from replay buffer (already numpy arrays)
            obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
            
            # Convert to tensors efficiently (zero-copy when possible)
            obs_tensor = torch.from_numpy(obs).to(self.device, non_blocking=True)
            actions_tensor = torch.from_numpy(actions).to(self.device, non_blocking=True)
            rewards_tensor = torch.from_numpy(rewards).to(self.device, non_blocking=True)
            next_obs_tensor = torch.from_numpy(next_obs).to(self.device, non_blocking=True)
            dones_tensor = torch.from_numpy(dones).to(self.device, non_blocking=True)
            
            # Compute target Q values (single forward pass for both Q networks)
            with torch.no_grad():
                next_logits = self.actor.get_logits(next_obs_tensor)
                next_probs = F.softmax(next_logits, dim=-1)
                next_log_probs = F.log_softmax(next_logits, dim=-1)
                
                q1_next = self.q1_target(next_obs_tensor)
                q2_next = self.q2_target(next_obs_tensor)
                q_next = torch.min(q1_next, q2_next)
                
                # Vectorized target computation
                target_q = rewards_tensor + (1.0 - dones_tensor) * self.gamma * (
                    torch.sum(next_probs * (q_next - self.alpha * next_log_probs), dim=1)
                )
            
            # Update Q networks (compute both in parallel)
            q1_values = self.q1(obs_tensor)
            q1_selected = q1_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            q1_loss = F.mse_loss(q1_selected, target_q, reduction='mean')
            
            q2_values = self.q2(obs_tensor)
            q2_selected = q2_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            q2_loss = F.mse_loss(q2_selected, target_q, reduction='mean')
            
            # Update Q1
            self.q1_optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()
            
            # Update Q2
            self.q2_optimizer.zero_grad(set_to_none=True)
            q2_loss.backward()
            self.q2_optimizer.step()
            
            # Update actor (reuse obs_tensor, detach Q values to avoid graph issues)
            obs_logits = self.actor.get_logits(obs_tensor)
            probs = F.softmax(obs_logits, dim=-1)
            log_probs = F.log_softmax(obs_logits, dim=-1)
            # Reuse Q values but detach them (we don't need gradients through Q networks for actor)
            q_vals = torch.min(q1_values.detach(), q2_values.detach())
            
            actor_loss = torch.sum(probs * (self.alpha * log_probs - q_vals), dim=1).mean()
            
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            
            losses["q1_loss"] += q1_loss.item()
            losses["q2_loss"] += q2_loss.item()
            losses["actor_loss"] += actor_loss.item()
            
            # Soft update target networks (only when needed)
            if self.num_updates % self.target_update_interval == 0:
                # Use in-place operations for efficiency
                with torch.no_grad():
                    for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                        target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
                    for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                        target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
            
            self.num_updates += 1
        
        # Average losses
        if self.gradient_steps > 0:
            losses["q1_loss"] /= self.gradient_steps
            losses["q2_loss"] /= self.gradient_steps
            losses["actor_loss"] /= self.gradient_steps
        
        return losses
    
    def learn(self, total_timesteps: int, callback=None):
        """Optimized training loop with reduced overhead."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Pre-compute callback check interval to avoid repeated checks
        callback_interval = TRACK_PROGRESS_INTERVAL if callback else float('inf')
        last_callback_step = 0
        
        for step in range(total_timesteps):
            # Select action (optimized condition check)
            if self.num_timesteps < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(obs, deterministic=False)
            
            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.replay_buffer.push(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            self.num_timesteps += 1
            
            # Update agent (optimized condition check)
            should_update = (self.num_timesteps >= self.learning_starts and 
                           self.num_timesteps % self.train_freq == 0)
            if should_update:
                self.update()
            
            # Reset environment if done
            if done:
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            
            # Callback (only check at intervals)
            if callback is not None and (self.num_timesteps - last_callback_step) >= callback_interval:
                callback(self)
                last_callback_step = self.num_timesteps
        
        return self
    
    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Predict action for given observation."""
        action = self.select_action(obs, deterministic=deterministic)
        return action, None
    
    def save(self, path: str):
        """Save the model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
        }, path)
    
    @classmethod
    def load(cls, path: str, env: gym.Env, **kwargs):
        """Load a saved model."""
        agent = cls(env, **kwargs)
        checkpoint = torch.load(path, map_location=agent.device)
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.q1.load_state_dict(checkpoint['q1'])
        agent.q2.load_state_dict(checkpoint['q2'])
        agent.q1_target.load_state_dict(checkpoint['q1_target'])
        agent.q2_target.load_state_dict(checkpoint['q2_target'])
        return agent


def evaluate_agent(env: BankEnv, agent, n_episodes: int = 100) -> Tuple[float, float, List[int], List[float]]:
    """Optimized evaluation with reduced overhead."""
    wins = 0
    ties = 0
    losses = 0
    total_score = 0.0
    episode_results = []
    episode_scores = []
    
    # Pre-allocate lists with estimated size
    episode_results = [0] * n_episodes
    episode_scores = [0.0] * n_episodes
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            if agent is not None:
                action, _ = agent.predict(obs, deterministic=True)
            else:
                current_score = info.get("current_score", 0)
                action = 1 if current_score > 50 else 0
            
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Get final scores (optimized condition checks)
        if terminated or truncated:
            final_scores = info.get("final_scores") or info.get("player_scores") or [0, 0]
            
            agent_score = final_scores[0]
            opponent_scores = final_scores[1:] if len(final_scores) > 1 else [0]
            max_opponent_score = max(opponent_scores) if opponent_scores else 0
            total_score += agent_score
            episode_scores[episode] = agent_score
            
            if agent_score > max_opponent_score:
                wins += 1
                episode_results[episode] = 1
            elif agent_score == max_opponent_score:
                ties += 1
                episode_results[episode] = 0
            else:
                losses += 1
                episode_results[episode] = -1
    
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


class ProgressTrackingCallback:
    """Callback to track training progress."""
    def __init__(self, eval_env: BankEnv, eval_episodes: int = 20):
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        self.timesteps = []
        self.win_rates = []
        self.avg_scores = []
        self.last_eval_step = 0
    
    def __call__(self, agent: SACDiscreteAgent):
        """Track progress during training."""
        if agent.num_timesteps - self.last_eval_step >= TRACK_PROGRESS_INTERVAL:
            self.last_eval_step = agent.num_timesteps
            try:
                win_rate, avg_score, _, _ = evaluate_agent(
                    self.eval_env, agent, self.eval_episodes
                )
                self.timesteps.append(agent.num_timesteps)
                self.win_rates.append(win_rate)
                self.avg_scores.append(avg_score)
                print(f"Progress at {agent.num_timesteps} timesteps: Win rate = {win_rate:.3f}, Avg score = {avg_score:.2f}")
            except Exception as e:
                print(f"Warning: Could not evaluate progress at {agent.num_timesteps}: {e}")


def train_agent():
    """Train the RL agent."""
    print("=" * 80)
    print("Starting SAC-Discrete Training (CleanRL-based)")
    print("=" * 80)
    print(f"Environment rounds: {ENV_ROUNDS}")
    print(f"Opponents: {len(ENV_OPPONENTS)} opponent(s)")
    print(f"Training timesteps: {TRAIN_TOTAL_TIMESTEPS}")
    print(f"Device: {DEVICE}")
    print()
    
    env = BankEnv(
        rounds=ENV_ROUNDS,
        opponents=ENV_OPPONENTS,
        max_round_length=ENV_MAX_ROUND_LENGTH,
        verbose=False,
    )
    
    agent = SACDiscreteAgent(
        env,
        learning_rate=SAC_DISCRETE_CONFIG["learning_rate"],
        buffer_size=SAC_DISCRETE_CONFIG["buffer_size"],
        batch_size=SAC_DISCRETE_CONFIG["batch_size"],
        tau=SAC_DISCRETE_CONFIG["tau"],
        gamma=SAC_DISCRETE_CONFIG["gamma"],
        alpha=SAC_DISCRETE_CONFIG["alpha"],
        learning_starts=SAC_DISCRETE_CONFIG["learning_starts"],
        target_update_interval=SAC_DISCRETE_CONFIG["target_update_interval"],
        gradient_steps=SAC_DISCRETE_CONFIG["gradient_steps"],
        train_freq=SAC_DISCRETE_CONFIG["train_freq"],
        device=DEVICE,
    )
    
    eval_env = BankEnv(
        rounds=ENV_ROUNDS,
        opponents=ENV_OPPONENTS,
        max_round_length=ENV_MAX_ROUND_LENGTH,
        verbose=False,
    )
    
    progress_callback = ProgressTrackingCallback(eval_env, PROGRESS_EVAL_EPISODES)
    
    print("Training with CleanRL-based SAC-Discrete...")
    agent.learn(TRAIN_TOTAL_TIMESTEPS, callback=progress_callback)
    
    training_win_rates = progress_callback.win_rates
    training_timesteps = progress_callback.timesteps
    training_avg_scores = progress_callback.avg_scores
    
    eval_env.close()
    
    if SAVE_MODEL:
        Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        agent.save(MODEL_SAVE_PATH)
        print(f"\nModel saved to {MODEL_SAVE_PATH}")
    
    env.close()
    
    return agent, training_win_rates, training_timesteps, training_avg_scores


def evaluate_trained_agent(agent):
    """Evaluate the trained agent."""
    print("\n" + "=" * 80)
    print("Evaluating Trained Agent")
    print("=" * 80)
    
    if agent is None:
        print("Warning: No trained model available. Using simple threshold policy for evaluation.")
    
    eval_env = BankEnv(
        rounds=ENV_ROUNDS,
        opponents=ENV_OPPONENTS,
        max_round_length=ENV_MAX_ROUND_LENGTH,
        verbose=False,
    )
    
    win_rate, avg_score, episode_results, episode_scores = evaluate_agent(
        eval_env, agent, EVAL_N_EPISODES
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
    fig.suptitle('RL Training Statistics (SAC-Discrete - CleanRL)', fontsize=16, fontweight='bold')
    
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
    agent, training_win_rates, training_timesteps, training_avg_scores = train_agent()
    
    post_train_win_rate, post_train_avg_score, episode_results, episode_scores = evaluate_trained_agent(agent)
    
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
