"""
PPO Agents Free-for-All Tournament

All 4 PPO agents (rl_round, rl_score, rl_sparse, rl_sparse_better) fight against
each other simultaneously in a 4-player free-for-all match.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from contextlib import redirect_stdout
import importlib.util

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO

# Add game_theory to path for imports
sys.path.insert(0, str(project_root / "game_theory"))
from bank import Bank
from players import Player

# ============================================================================
# CONFIGURATION
# ============================================================================

# Number of games to run
N_GAMES = 10000

# Number of rounds per game
ROUNDS = 10

# Model paths
MODEL_PATHS = {
    "rl_round": project_root / "big_rl" / "rl_round" / "RL_data" / "rl_bank_model_ppo_multi.zip",
    "rl_score": project_root / "big_rl" / "rl_score" / "RL_data" / "rl_bank_model_ppo_multi.zip",
    "rl_sparse": project_root / "big_rl" / "rl_sparse" / "RL_data" / "rl_bank_model_ppo_multi.zip",
    "rl_sparse_better": project_root / "big_rl" / "rl_sparse_better" / "RL_data" / "rl_bank_model_ppo_multi.zip",
}

# ============================================================================
# RL MODEL PLAYER WRAPPER
# ============================================================================

class RLModelPlayer(Player):
    """A Player wrapper that uses an RL model to make decisions."""
    
    def __init__(self, model, name: str = None, n_players: int = 4):
        """
        Initialize RL model player.
        
        Args:
            model: The trained PPO model
            name: Name of the player
            n_players: Total number of players in the game (for observation space)
        """
        super().__init__(name=name)
        self.model = model
        self.n_players = n_players
        self.score_scale = 5000.0  # Same normalization as in BankEnv
        
        # Get expected observation space size from the model
        obs_space = model.observation_space
        if hasattr(obs_space, 'shape'):
            self.expected_obs_size = obs_space.shape[0]
        else:
            # Fallback: assume it was trained with 1 opponent (2 players total)
            self.expected_obs_size = 6  # 1 + 1 + 2 + 2
    
    def state_to_observation(self, state: dict) -> np.ndarray:
        """
        Convert game state dictionary to observation array for the RL model.
        
        Observation format: [current_score, rounds_remaining, player_0_score, ..., 
                            player_n_score, player_0_in, ..., player_n_in]
        
        The observation is padded/truncated to match what the model expects.
        """
        # Build full observation for current game
        full_obs = np.zeros(1 + 1 + self.n_players + self.n_players, dtype=np.float32)
        
        # Normalize current_score
        full_obs[0] = float(state["current_score"]) / self.score_scale
        
        # Normalize rounds_remaining
        full_obs[1] = float(state["rounds_remaining"]) / float(max(ROUNDS, 1))
        
        # Normalize player scores
        player_scores = state["player_scores"]
        for i in range(self.n_players):
            if i < len(player_scores):
                full_obs[2 + i] = float(player_scores[i]) / self.score_scale
        
        # Players in (binary)
        players_in = state["players_in"]
        for i in range(self.n_players):
            if i < len(players_in):
                full_obs[2 + self.n_players + i] = float(players_in[i])
        
        # Adjust to match model's expected size
        if len(full_obs) == self.expected_obs_size:
            return full_obs
        elif len(full_obs) > self.expected_obs_size:
            # Truncate: take first expected_obs_size elements
            # This means we're dropping information about extra players
            return full_obs[:self.expected_obs_size]
        else:
            # Pad: add zeros at the end
            obs = np.zeros(self.expected_obs_size, dtype=np.float32)
            obs[:len(full_obs)] = full_obs
            return obs
    
    def decide_action(self, state: dict) -> str:
        """
        Use the RL model to decide action.
        
        Args:
            state: Game state dictionary
            
        Returns:
            "bank" or "roll"
        """
        # Convert state to observation
        obs = self.state_to_observation(state)
        
        # Get action from model (0 = roll, 1 = bank)
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Convert to string
        return "bank" if action == 1 else "roll"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_env_class(folder_name):
    """Dynamically load the BankEnv class from a specific folder."""
    folder_path = project_root / "big_rl" / folder_name
    folder_path_str = str(folder_path)
    
    # Remove if already in path to avoid conflicts
    if folder_path_str in sys.path:
        sys.path.remove(folder_path_str)
    sys.path.insert(0, folder_path_str)
    
    # Import the BankEnv from this specific folder
    spec = importlib.util.spec_from_file_location(
        f"bank_gym_{folder_name}",
        folder_path / "bank_gym.py"
    )
    bank_gym_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bank_gym_module)
    
    return bank_gym_module.BankEnv

def load_model(model_path, folder_name):
    """Load a PPO model from the given path using the appropriate environment."""
    BankEnvClass = load_env_class(folder_name)
    
    # Create template environment for loading model
    # Use a dummy opponent for initialization
    from game_theory.players import ThersholdPlayer
    template_opponents = [ThersholdPlayer(threshold=100)]
    template_env = BankEnvClass(
        rounds=ROUNDS,
        opponents=template_opponents,
        max_round_length=100,
        verbose=False
    )
    
    # Load model (suppress output)
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull):
            model = PPO.load(str(model_path), env=template_env)
    
    template_env.close()
    return model

def run_tournament_game(players: list[RLModelPlayer], seed=None) -> dict:
    """
    Run a single tournament game with all players.
    
    Returns:
        Dictionary with final scores and winner info
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create game
    game = Bank(rounds=ROUNDS, players=players, verbose=False)
    game.play_game()
    
    # Get final scores
    final_scores = game.player_scores
    
    # Determine winner(s) - could be ties
    max_score = max(final_scores)
    winners = [i for i, score in enumerate(final_scores) if score == max_score]
    
    return {
        'final_scores': final_scores,
        'winners': winners,
        'max_score': max_score,
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function to run the tournament."""
    print("=" * 80)
    print("PPO Agents Free-for-All Tournament")
    print("=" * 80)
    print(f"4 agents fighting simultaneously")
    print(f"Running {N_GAMES} games with {ROUNDS} rounds each\n")
    
    # Check all models exist
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            print(f"Error: Model not found at {path}")
            return
    
    # Load all models
    print("Loading models...")
    models = {}
    
    for folder_name, model_path in MODEL_PATHS.items():
        print(f"  Loading {folder_name}...", end=" ", flush=True)
        try:
            model = load_model(model_path, folder_name)
            models[folder_name] = model
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print()
    
    # Create RL players
    print("Creating RL players...")
    agent_names = list(MODEL_PATHS.keys())
    players = []
    for agent_name in agent_names:
        player = RLModelPlayer(
            model=models[agent_name],
            name=agent_name,
            n_players=4
        )
        players.append(player)
        print(f"  {agent_name} ready")
    
    print()
    
    # Run tournament
    print(f"Running tournament ({N_GAMES} games)...")
    wins = {name: 0 for name in agent_names}
    ties = {name: 0 for name in agent_names}
    total_scores = {name: 0.0 for name in agent_names}
    
    for game_num in range(N_GAMES):
        if (game_num + 1) % 1000 == 0:
            print(f"  Progress: {game_num + 1}/{N_GAMES} games...", flush=True)
        
        # Run game
        result = run_tournament_game(players, seed=None)
        
        # Update statistics
        final_scores = result['final_scores']
        winners = result['winners']
        
        for i, agent_name in enumerate(agent_names):
            total_scores[agent_name] += final_scores[i]
            
            if i in winners:
                if len(winners) == 1:
                    wins[agent_name] += 1
                else:
                    ties[agent_name] += 1 / len(winners)  # Split tie credit
    
    print()
    
    # Calculate statistics
    win_rates = {name: wins[name] / N_GAMES for name in agent_names}
    tie_rates = {name: ties[name] / N_GAMES for name in agent_names}
    avg_scores = {name: total_scores[name] / N_GAMES for name in agent_names}
    
    # Print results
    print("=" * 80)
    print("Tournament Results")
    print("=" * 80)
    print(f"\n{'Agent':<20s} {'Wins':<8s} {'Ties':<12s} {'Win Rate':<12s} {'Avg Score':<12s}")
    print("-" * 80)
    
    for agent_name in agent_names:
        print(f"{agent_name:<20s} {wins[agent_name]:<8d} {ties[agent_name]:<12.2f} "
              f"{win_rates[agent_name]:<12.3f} {avg_scores[agent_name]:<12.2f}")
    
    # Create bar chart
    print("\nGenerating bar chart...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Win rates
    win_rate_values = [win_rates[name] for name in agent_names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars1 = ax1.bar(agent_names, win_rate_values, color=colors)
    ax1.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
    ax1.set_title('Win Rates (4-Player Free-for-All)', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, max(win_rate_values) * 1.2 if max(win_rate_values) > 0 else 1])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, linewidth=1, 
                label='Expected (25% for equal players)')
    ax1.legend()
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars1, win_rate_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.3f}\n({wins[agent_names[i]]} wins)',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Subplot 2: Average scores
    avg_score_values = [avg_scores[name] for name in agent_names]
    
    bars2 = ax2.bar(agent_names, avg_score_values, color=colors)
    ax2.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax2.set_title('Average Scores (4-Player Free-for-All)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars2, avg_score_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = project_root / "player_arena"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "ppo_agents_tournament.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    print("\nTournament complete!")

if __name__ == "__main__":
    main()

