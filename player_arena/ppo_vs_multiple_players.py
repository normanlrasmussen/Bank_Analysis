"""
PPO Agent vs Multiple Opponents Simultaneously

This script allows you to test the original PPO agent against multiple opponents
at the same time. You can configure a list of opponents (e.g., 3 threshold agents)
and run games with all of them playing together.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
from contextlib import redirect_stdout

# Add parent directory to path to import rl_og modules and game_theory
project_root = Path(__file__).resolve().parent.parent

# Add project root to path (for game_theory module)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add rl_og to path (for rl_og modules)
rl_og_path = str(project_root / "rl_og")
if rl_og_path not in sys.path:
    sys.path.insert(0, rl_og_path)

# Also add parent directory in case we're running from a different location
parent_dir = str(project_root.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import, and if game_theory is not found, ensure project_root is in path
try:
    from stable_baselines3 import PPO
    from rl_og.bank_gym import BankEnv
    from rl_og.players import (
        ThersholdPlayer, GreedyPlayer, GreedyPlayerK, SesquaGreedyPlayer,
        ProbabilisticPlayer, AntiGreedyPlayer, TrollPlayer
    )
except ModuleNotFoundError as e:
    if 'game_theory' in str(e):
        # Ensure project_root is definitely in the path
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        # Try importing again
        from stable_baselines3 import PPO
        from rl_og.bank_gym import BankEnv
        from rl_og.players import (
            ThersholdPlayer, GreedyPlayer, GreedyPlayerK, SesquaGreedyPlayer,
            ProbabilisticPlayer, AntiGreedyPlayer, TrollPlayer
        )
    else:
        raise


# ============================================================================
# CONFIGURATION - Edit these to choose your opponents
# ============================================================================

# Number of games to run
N_GAMES = 1000

# Number of rounds per game
ROUNDS = 10

# List of opponents to play against simultaneously
# You can add multiple of the same type (e.g., 3 threshold agents)
# Format: (name, player_instance)
OPPONENTS = [
    ("Threshold(80)", ThersholdPlayer(threshold=80)),
    ("Threshold(120)", ThersholdPlayer(threshold=120)),
    ("Threshold(500)", ThersholdPlayer(threshold=500)),
    ("SesquaGreedy", SesquaGreedyPlayer()),
]

# PPO Model path (original PPO model)
PPO_MODEL_PATH = project_root / "rl_og" / "RL_data" / "rl_bank_model_ppo.zip"

# Image save path
image_path_save = project_root / "player_arena"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_obs_to_6dim(obs_10dim):
    """
    Convert 10-dimensional observation (4 players) to 6-dimensional (2 players).
    
    10-dim format: [score, rounds, p0, p1, p2, p3, in0, in1, in2, in3]
    6-dim format: [score, rounds, p0, max(p1,p2,p3), in0, any(in1,in2,in3)]
    """
    obs_6dim = np.zeros(6, dtype=np.float32)
    
    # Copy score and rounds (first 2 elements)
    obs_6dim[0] = obs_10dim[0]
    obs_6dim[1] = obs_10dim[1]
    
    # Agent score (p0)
    obs_6dim[2] = obs_10dim[2]
    
    # Max opponent score (max of p1, p2, p3)
    obs_6dim[3] = max(obs_10dim[3], obs_10dim[4], obs_10dim[5]) if len(obs_10dim) > 5 else max(obs_10dim[3], obs_10dim[4])
    
    # Agent in
    obs_6dim[4] = obs_10dim[6]
    
    # Any opponent in
    obs_6dim[5] = max(obs_10dim[7], obs_10dim[8], obs_10dim[9]) if len(obs_10dim) > 9 else max(obs_10dim[7], obs_10dim[8])
    
    return obs_6dim


def run_game(ppo_model, opponents, env_template, seed=None):
    """
    Run a single game with PPO agent against multiple opponents.
    
    Returns:
        Tuple of (ppo_score, opponent_scores, total_reward)
    """
    # Create fresh copies of opponents
    import copy
    fresh_opponents = [copy.deepcopy(opp) for _, opp in opponents]
    
    env = BankEnv(
        rounds=env_template.rounds,
        opponents=fresh_opponents,
        max_round_length=env_template.max_round_length,
        verbose=False
    )
    
    obs_10dim, info = env.reset(seed=seed)
    # Convert 10-dim observation to 6-dim for PPO model (trained with 1 opponent)
    obs_6dim = convert_obs_to_6dim(obs_10dim)
    
    terminated = False
    truncated = False
    total_reward = 0.0
    
    while not (terminated or truncated):
        # Use 6-dim observation for PPO model prediction
        action, _ = ppo_model.predict(obs_6dim, deterministic=True)
        obs_10dim, reward, terminated, truncated, info = env.step(action)
        # Convert next observation to 6-dim
        obs_6dim = convert_obs_to_6dim(obs_10dim)
        total_reward += reward
    
    # Get final scores
    final_scores = info.get('final_scores', info.get('player_scores', []))
    ppo_score = final_scores[0] if len(final_scores) > 0 else 0
    opponent_scores = final_scores[1:] if len(final_scores) > 1 else []
    
    env.close()
    
    return ppo_score, opponent_scores, total_reward


def evaluate_ppo_vs_multiple_opponents(ppo_model, opponents, env_template, n_games=1000):
    """
    Evaluate PPO agent against multiple opponents simultaneously.
    
    Returns:
        Dictionary with statistics
    """
    n_opponents = len(opponents)
    results = {
        'ppo_wins': 0,
        'ppo_ties': 0,
        'ppo_losses': 0,
        'ppo_scores': [],
        'opponent_scores': [[] for _ in range(n_opponents)],
        'opponent_names': [name for name, _ in opponents],
    }
    
    for game in range(n_games):
        if (game + 1) % 100 == 0:
            print(f"  Progress: {game + 1}/{n_games} games...", flush=True)
        
        ppo_score, opponent_scores, _ = run_game(ppo_model, opponents, env_template, seed=None)
        
        results['ppo_scores'].append(ppo_score)
        for i, score in enumerate(opponent_scores):
            if i < n_opponents:
                results['opponent_scores'][i].append(score)
        
        # Determine win/loss/tie (PPO wins if it beats all opponents)
        max_opponent_score = max(opponent_scores) if opponent_scores else 0
        
        if ppo_score > max_opponent_score:
            results['ppo_wins'] += 1
        elif ppo_score == max_opponent_score:
            results['ppo_ties'] += 1
        else:
            results['ppo_losses'] += 1
    
    # Calculate statistics
    results['ppo_avg_score'] = np.mean(results['ppo_scores'])
    results['ppo_std_score'] = np.std(results['ppo_scores'])
    results['opponent_avg_scores'] = [np.mean(scores) if scores else 0.0 for scores in results['opponent_scores']]
    results['opponent_std_scores'] = [np.std(scores) if scores else 0.0 for scores in results['opponent_scores']]
    results['win_rate'] = (results['ppo_wins'] + 0.5 * results['ppo_ties']) / n_games
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function to run PPO vs multiple opponents."""
    print("=" * 80)
    print("PPO Agent vs Multiple Opponents Simultaneously")
    print("=" * 80)
    
    # Check if PPO model exists
    if not PPO_MODEL_PATH.exists():
        print(f"Error: PPO model not found at {PPO_MODEL_PATH}")
        print("Please make sure the model file exists.")
        return
    
    # Validate opponents list
    if len(OPPONENTS) == 0:
        print("Error: No opponents specified. Please add opponents to the OPPONENTS list.")
        return
    
    # Create template environment for loading model
    template_opponents = [ThersholdPlayer(threshold=100)]
    template_env = BankEnv(
        rounds=ROUNDS,
        opponents=template_opponents,
        max_round_length=100,
        verbose=False
    )
    
    # Load PPO model (suppress stable-baselines3 print statements)
    print(f"\nLoading PPO model from {PPO_MODEL_PATH}...")
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull):
            ppo_model = PPO.load(str(PPO_MODEL_PATH), env=template_env)
    print("✓ PPO model loaded successfully")
    
    template_env.close()
    
    # Create game environment template
    game_env = BankEnv(
        rounds=ROUNDS,
        opponents=[],  # Will be set per game
        max_round_length=100,
        verbose=False
    )
    
    print(f"\nOpponents: {', '.join([name for name, _ in OPPONENTS])}")
    print(f"Total opponents: {len(OPPONENTS)}")
    print(f"Running {N_GAMES} games with PPO vs all opponents simultaneously...\n")
    
    # Run evaluation
    results = evaluate_ppo_vs_multiple_opponents(
        ppo_model, OPPONENTS, game_env, n_games=N_GAMES
    )
    
    game_env.close()
    
    # Print results
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"\nGames played: {N_GAMES}")
    print(f"Win Rate: {results['win_rate']:.3f} ({results['win_rate']*100:.1f}%)")
    print(f"Wins: {results['ppo_wins']}, Ties: {results['ppo_ties']}, Losses: {results['ppo_losses']}")
    print(f"\nPPO Average Score: {results['ppo_avg_score']:.2f} ± {results['ppo_std_score']:.2f}")
    
    print(f"\nOpponent Scores:")
    for i, (name, avg_score, std_score) in enumerate(zip(
        results['opponent_names'],
        results['opponent_avg_scores'],
        results['opponent_std_scores']
    )):
        print(f"  {name:<25s}: {avg_score:>8.2f} ± {std_score:>6.2f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Win/Loss/Tie pie chart
    ax1 = axes[0, 0]
    labels = ['Wins', 'Ties', 'Losses']
    sizes = [results['ppo_wins'], results['ppo_ties'], results['ppo_losses']]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('PPO Game Outcomes')
    
    # 2. Score comparison bar chart
    ax2 = axes[0, 1]
    all_names = ['PPO'] + results['opponent_names']
    all_avgs = [results['ppo_avg_score']] + results['opponent_avg_scores']
    all_stds = [results['ppo_std_score']] + results['opponent_std_scores']
    
    bars = ax2.barh(all_names, all_avgs, xerr=all_stds, capsize=5, color=['steelblue'] + ['orange'] * len(results['opponent_names']))
    ax2.set_xlabel('Average Score')
    ax2.set_title('Average Scores: PPO vs Opponents')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, avg) in enumerate(zip(bars, all_avgs)):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{avg:.1f}',
                ha='left', va='center', fontweight='bold')
    
    # 3. Score distribution histogram
    ax3 = axes[1, 0]
    ax3.hist(results['ppo_scores'], bins=30, alpha=0.7, color='steelblue', label='PPO', edgecolor='black')
    ax3.axvline(results['ppo_avg_score'], color='red', linestyle='--', linewidth=2, label=f'PPO Mean ({results["ppo_avg_score"]:.1f})')
    ax3.set_xlabel('Final Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('PPO Score Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Opponent score distributions
    ax4 = axes[1, 1]
    for i, (name, scores) in enumerate(zip(results['opponent_names'], results['opponent_scores'])):
        if scores:
            ax4.hist(scores, bins=20, alpha=0.5, label=name, edgecolor='black')
    ax4.set_xlabel('Final Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Opponent Score Distributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Ensure the directory exists and save the figure
    image_path_save.mkdir(parents=True, exist_ok=True)
    save_path = image_path_save / "ppo_vs_multiple_players.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    
    print("\nTournament complete!")


if __name__ == "__main__":
    main()

