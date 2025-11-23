"""
PPO Agent vs Various Players

This script allows you to test the original PPO agent against various player types
that you can choose. You can configure the opponents and number of games to run.
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

# Opponents to test against (you can add/remove/modify these)
OPPONENTS = [
    ("Threshold(50)", ThersholdPlayer(threshold=50)),
    ("Threshold(100)", ThersholdPlayer(threshold=100)),
    ("Threshold(150)", ThersholdPlayer(threshold=150)),
    ("Threshold(200)", ThersholdPlayer(threshold=200)),
    ("Threshold(500)", ThersholdPlayer(threshold=500)),
    ("Threshold(1000)", ThersholdPlayer(threshold=1000)),
    ("GreedyPlayer", GreedyPlayer()),
    ("SesquaGreedy", SesquaGreedyPlayer()),
    ("Probabilistic(0.2)", ProbabilisticPlayer(probability=0.2)),
    ("Probabilistic(0.5)", ProbabilisticPlayer(probability=0.5)),
    ("Probabilistic(0.11)", ProbabilisticPlayer(probability=0.11)),
]

# PPO Model path (original PPO model)
PPO_MODEL_PATH = project_root / "rl_og" / "RL_data" / "rl_bank_model_ppo.zip"

image_path_save = project_root / "player_arena" 


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_game(ppo_model, opponents, env_template, seed=None):
    """
    Run a single game with PPO agent against opponents.
    
    Returns:
        Tuple of (ppo_score, opponent_scores, total_reward)
    """
    # Create fresh copies of opponents
    import copy
    fresh_opponents = [copy.deepcopy(opp) for opp in opponents]
    
    env = BankEnv(
        rounds=env_template.rounds,
        opponents=fresh_opponents,
        max_round_length=env_template.max_round_length,
        verbose=False
    )
    
    obs, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    
    while not (terminated or truncated):
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    
    # Get final scores
    final_scores = info.get('final_scores', info.get('player_scores', []))
    ppo_score = final_scores[0] if len(final_scores) > 0 else 0
    opponent_scores = final_scores[1:] if len(final_scores) > 1 else []
    
    env.close()
    
    return ppo_score, opponent_scores, total_reward


def evaluate_ppo_vs_opponents(ppo_model, opponents, opponent_names, env_template, n_games=1000):
    """
    Evaluate PPO agent against a set of opponents.
    
    Returns:
        Dictionary with statistics for each opponent
    """
    n_opponents = len(opponents)
    results = {
        'ppo_wins': 0,
        'ppo_ties': 0,
        'ppo_losses': 0,
        'ppo_scores': [],
        'opponent_scores': [[] for _ in range(n_opponents)],
        'ppo_avg_score': 0.0,
        'opponent_avg_scores': [0.0] * n_opponents,
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
    
    # Calculate averages
    results['ppo_avg_score'] = np.mean(results['ppo_scores'])
    for i in range(n_opponents):
        if results['opponent_scores'][i]:
            results['opponent_avg_scores'][i] = np.mean(results['opponent_scores'][i])
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function to run PPO vs various players."""
    print("=" * 80)
    print("PPO Agent vs Various Players")
    print("=" * 80)
    
    # Check if PPO model exists
    if not PPO_MODEL_PATH.exists():
        print(f"Error: PPO model not found at {PPO_MODEL_PATH}")
        print("Please make sure the model file exists.")
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
        opponents=[],  # Will be set per opponent
        max_round_length=100,
        verbose=False
    )
    
    print(f"\nTesting PPO agent against {len(OPPONENTS)} opponent types...")
    print(f"Running {N_GAMES} games per opponent type\n")
    
    # Test against each opponent type
    all_results = {}
    
    for opp_name, opp_player in OPPONENTS:
        print(f"Testing vs {opp_name}...")
        opponents = [opp_player]
        opponent_names = [opp_name]
        
        results = evaluate_ppo_vs_opponents(
            ppo_model, opponents, opponent_names, game_env, n_games=N_GAMES
        )
        
        all_results[opp_name] = results
        
        win_rate = (results['ppo_wins'] + 0.5 * results['ppo_ties']) / N_GAMES
        print(f"  Win Rate: {win_rate:.3f} ({results['ppo_wins']}W-{results['ppo_ties']}T-{results['ppo_losses']}L)")
        print(f"  Avg Scores: PPO={results['ppo_avg_score']:.2f}, {opp_name}={results['opponent_avg_scores'][0]:.2f}")
        print()
    
    game_env.close()
    
    # Print summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\n{'Opponent':<25s} {'Win Rate':<12s} {'PPO Avg':<12s} {'Opp Avg':<12s} {'W-T-L':<15s}")
    print("-" * 80)
    
    for opp_name, results in all_results.items():
        win_rate = (results['ppo_wins'] + 0.5 * results['ppo_ties']) / N_GAMES
        opp_avg = results['opponent_avg_scores'][0] if results['opponent_avg_scores'] else 0.0
        print(f"{opp_name:<25s} {win_rate:<12.3f} {results['ppo_avg_score']:<12.2f} "
              f"{opp_avg:<12.2f} {results['ppo_wins']}-{results['ppo_ties']}-{results['ppo_losses']}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Win rate comparison
    ax1 = axes[0]
    opp_names = list(all_results.keys())
    win_rates = [(all_results[name]['ppo_wins'] + 0.5 * all_results[name]['ppo_ties']) / N_GAMES 
                 for name in opp_names]
    
    bars = ax1.barh(opp_names, win_rates, color='steelblue')
    ax1.set_xlabel('Win Rate')
    ax1.set_title('PPO Win Rate vs Different Opponents')
    ax1.set_xlim([0, 1])
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, win_rates)):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f'{rate:.3f}',
                ha='left', va='center', fontweight='bold')
    
    # Average score comparison
    ax2 = axes[1]
    ppo_avgs = [all_results[name]['ppo_avg_score'] for name in opp_names]
    opp_avgs = [all_results[name]['opponent_avg_scores'][0] if all_results[name]['opponent_avg_scores'] else 0.0 
                for name in opp_names]
    
    x = np.arange(len(opp_names))
    width = 0.35
    
    bars1 = ax2.barh(x - width/2, ppo_avgs, width, label='PPO', color='steelblue')
    bars2 = ax2.barh(x + width/2, opp_avgs, width, label='Opponent', color='orange')
    
    ax2.set_yticks(x)
    ax2.set_yticklabels(opp_names)
    ax2.set_xlabel('Average Score')
    ax2.set_title('Average Scores: PPO vs Opponents')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Ensure the directory exists
    image_path_save.mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    save_path = image_path_save / "ppo_vs_players_win_rate.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    
    print("\nTournament complete!")


if __name__ == "__main__":
    main()

