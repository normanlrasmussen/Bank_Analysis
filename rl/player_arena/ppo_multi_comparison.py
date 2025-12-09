"""
Compare PPO Multi Agents from 4 big_rl folders against benchmarks

This script loads the ppo_multi agents from:
- rl_round
- rl_score
- rl_sparse
- rl_sparse_better

And tests them against benchmark players, plotting results on a single bar chart.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from contextlib import redirect_stdout

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO

# Import players from rl_round (they should be compatible across folders)
sys.path.insert(0, str(project_root / "big_rl" / "rl_round"))
from players import (
    ThersholdPlayer, GreedyPlayer, GreedyPlayerK, SesquaGreedyPlayer,
    ProbabilisticPlayer, AntiGreedyPlayer, TrollPlayer
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Number of games to run per agent vs each benchmark
N_GAMES = 500

# Number of rounds per game
ROUNDS = 10

# Model paths
MODEL_PATHS = {
    "rl_round": project_root / "big_rl" / "rl_round" / "RL_data" / "rl_bank_model_ppo_multi.zip",
    "rl_score": project_root / "big_rl" / "rl_score" / "RL_data" / "rl_bank_model_ppo_multi.zip",
    "rl_sparse": project_root / "big_rl" / "rl_sparse" / "RL_data" / "rl_bank_model_ppo_multi.zip",
    "rl_sparse_better": project_root / "big_rl" / "rl_sparse_better" / "RL_data" / "rl_bank_model_ppo_multi.zip",
}

# Cache for BankEnv classes
ENV_CLASSES = {}

# Benchmark opponents
BENCHMARKS = [
    ("Threshold(50)", lambda: ThersholdPlayer(threshold=50)),
    ("Threshold(100)", lambda: ThersholdPlayer(threshold=100)),
    ("Threshold(150)", lambda: ThersholdPlayer(threshold=150)),
    ("Threshold(200)", lambda: ThersholdPlayer(threshold=200)),
    ("GreedyPlayer", lambda: GreedyPlayer()),
    ("SesquaGreedy", lambda: SesquaGreedyPlayer()),
    ("Probabilistic(0.2)", lambda: ProbabilisticPlayer(probability=0.2)),
    ("Probabilistic(0.5)", lambda: ProbabilisticPlayer(probability=0.5)),
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_env_class(folder_name):
    """Dynamically load the BankEnv class from a specific folder."""
    if folder_name in ENV_CLASSES:
        return ENV_CLASSES[folder_name]
    
    # Add folder to path (at the beginning to prioritize)
    folder_path = project_root / "big_rl" / folder_name
    folder_path_str = str(folder_path)
    
    # Remove if already in path to avoid conflicts
    if folder_path_str in sys.path:
        sys.path.remove(folder_path_str)
    sys.path.insert(0, folder_path_str)
    
    # Import the BankEnv from this specific folder
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"bank_gym_{folder_name}",
        folder_path / "bank_gym.py"
    )
    bank_gym_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bank_gym_module)
    
    BankEnvClass = bank_gym_module.BankEnv
    ENV_CLASSES[folder_name] = BankEnvClass
    return BankEnvClass

def load_model(model_path, folder_name):
    """Load a PPO model from the given path using the appropriate environment."""
    BankEnvClass = load_env_class(folder_name)
    
    # Create template environment for loading model
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
    return model, BankEnvClass

def run_game(model, opponents, BankEnvClass, seed=None):
    """Run a single game with the model against opponents."""
    fresh_opponents = [copy.deepcopy(opp) for opp in opponents]
    
    env = BankEnvClass(
        rounds=ROUNDS,
        opponents=fresh_opponents,
        max_round_length=100,
        verbose=False
    )
    
    obs, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    
    # Get final scores
    final_scores = info.get('final_scores', info.get('player_scores', []))
    model_score = final_scores[0] if len(final_scores) > 0 else 0
    opponent_scores = final_scores[1:] if len(final_scores) > 1 else []
    
    env.close()
    
    return model_score, opponent_scores, total_reward

def evaluate_model_vs_benchmarks(model, BankEnvClass, n_games=N_GAMES):
    """Evaluate a model against all benchmarks."""
    results = {}
    
    for bench_name, bench_factory in BENCHMARKS:
        print(f"  Testing vs {bench_name}...", end=" ", flush=True)
        
        wins = 0
        ties = 0
        losses = 0
        model_scores = []
        bench_scores = []
        
        for game in range(n_games):
            opponent = bench_factory()
            model_score, opponent_scores, _ = run_game(
                model, [opponent], BankEnvClass, seed=None
            )
            
            model_scores.append(model_score)
            bench_score = opponent_scores[0] if opponent_scores else 0
            bench_scores.append(bench_score)
            
            if model_score > bench_score:
                wins += 1
            elif model_score == bench_score:
                ties += 1
            else:
                losses += 1
        
        win_rate = (wins + 0.5 * ties) / n_games
        avg_model_score = np.mean(model_scores)
        avg_bench_score = np.mean(bench_scores)
        
        results[bench_name] = {
            'win_rate': win_rate,
            'wins': wins,
            'ties': ties,
            'losses': losses,
            'avg_model_score': avg_model_score,
            'avg_bench_score': avg_bench_score,
        }
        
        print(f"Win Rate: {win_rate:.3f}")
    
    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function to compare all PPO multi agents."""
    print("=" * 80)
    print("PPO Multi Agent Comparison")
    print("=" * 80)
    print(f"Testing {len(MODEL_PATHS)} agents against {len(BENCHMARKS)} benchmarks")
    print(f"Running {N_GAMES} games per agent-benchmark pair\n")
    
    # Check all models exist
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            print(f"Error: Model not found at {path}")
            return
    
    # Load all models
    print("Loading models...")
    models = {}
    env_classes = {}
    
    for folder_name, model_path in MODEL_PATHS.items():
        print(f"  Loading {folder_name}...", end=" ", flush=True)
        try:
            model, BankEnvClass = load_model(model_path, folder_name)
            models[folder_name] = model
            env_classes[folder_name] = BankEnvClass
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
            return
    
    print()
    
    # Evaluate each model
    all_results = {}
    
    for folder_name, model in models.items():
        print(f"Evaluating {folder_name}...")
        BankEnvClass = env_classes[folder_name]
        results = evaluate_model_vs_benchmarks(model, BankEnvClass, n_games=N_GAMES)
        all_results[folder_name] = results
        print()
    
    # Print summary table
    print("=" * 80)
    print("Summary - Win Rates")
    print("=" * 80)
    
    # Header
    header = f"{'Benchmark':<20s}"
    for folder_name in MODEL_PATHS.keys():
        header += f" {folder_name:<15s}"
    print(header)
    print("-" * 80)
    
    # Rows
    for bench_name, _ in BENCHMARKS:
        row = f"{bench_name:<20s}"
        for folder_name in MODEL_PATHS.keys():
            win_rate = all_results[folder_name][bench_name]['win_rate']
            row += f" {win_rate:<15.3f}"
        print(row)
    
    # Create bar chart
    print("\nGenerating bar chart...")
    
    # Prepare data for plotting
    benchmark_names = [name for name, _ in BENCHMARKS]
    agent_names = list(MODEL_PATHS.keys())
    
    # Win rates matrix: [benchmark][agent]
    win_rates = np.array([
        [all_results[agent][bench]['win_rate'] 
         for agent in agent_names]
        for bench in benchmark_names
    ])
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(benchmark_names))
    width = 0.2  # Width of bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Distinct colors
    
    bars = []
    for i, agent_name in enumerate(agent_names):
        bars.append(ax.bar(
            x + i * width,
            win_rates[:, i],
            width,
            label=agent_name,
            color=colors[i % len(colors)]
        ))
    
    ax.set_xlabel('Benchmark Player', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
    ax.set_title('PPO Multi Agents vs Benchmarks - Win Rate Comparison', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(agent_names) - 1) / 2)
    ax.set_xticklabels(benchmark_names, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add value labels on bars
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = project_root / "player_arena"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "ppo_multi_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    print("\nComparison complete!")

if __name__ == "__main__":
    main()

