"""
PPO Statistics Collection Script

Tests 4 multi PPO models and 3 troll PPO models against 5 opponent types,
runs 100 games per matchup, and collects detailed statistics.
"""

import sys
from pathlib import Path
import numpy as np
import copy
import os
from contextlib import redirect_stdout
import importlib.util
from typing import Dict, List, Tuple, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO

# Add game_theory to path for imports
sys.path.insert(0, str(project_root / "game_theory"))
from bank import Bank
from players import Player, ThersholdPlayer, GreedyPlayer, SesquaGreedyPlayer, TrollPlayer

# ============================================================================
# CONFIGURATION
# ============================================================================

# Number of games to run per matchup
N_GAMES = 1000

# Number of rounds per game
ROUNDS = 10

# Model paths - Multi PPO models (4)
MULTI_MODEL_PATHS = {
    "rl_round_multi": {
        "path": project_root / "big_rl" / "rl_round" / "RL_data" / "rl_bank_model_ppo_multi.zip",
        "folder": "rl_round"
    },
    "rl_score_multi": {
        "path": project_root / "big_rl" / "rl_score" / "RL_data" / "rl_bank_model_ppo_multi.zip",
        "folder": "rl_score"
    },
    "rl_sparse_multi": {
        "path": project_root / "big_rl" / "rl_sparse" / "RL_data" / "rl_bank_model_ppo_multi.zip",
        "folder": "rl_sparse"
    },
    "rl_sparse_better_multi": {
        "path": project_root / "big_rl" / "rl_sparse_better" / "RL_data" / "rl_bank_model_ppo_multi.zip",
        "folder": "rl_sparse_better"
    },
}

# Model paths - Troll PPO models (3)
TROLL_MODEL_PATHS = {
    "rl_round_troll": {
        "path": project_root / "big_rl" / "rl_round" / "RL_data" / "rl_bank_model_ppo_troll.zip",
        "folder": "rl_round"
    },
    "rl_sparse_troll": {
        "path": project_root / "big_rl" / "rl_sparse" / "RL_data" / "rl_bank_model_ppo_troll.zip",
        "folder": "rl_sparse"
    },
    "rl_sparse_better_troll": {
        "path": project_root / "big_rl" / "rl_sparse_better" / "RL_data" / "rl_bank_model_ppo_troll.zip",
        "folder": "rl_sparse_better"
    },
}

# Opponent types
OPPONENTS = {
    "100_threshold": ThersholdPlayer(threshold=100),
    "180_threshold": ThersholdPlayer(threshold=180),
    "greedy": GreedyPlayer(),
    "sesquagreedy": SesquaGreedyPlayer(),
    "troll": TrollPlayer(),
}

# ============================================================================
# STATISTICS TRACKING BANK WRAPPER
# ============================================================================

class StatisticsBank(Bank):
    """Extended Bank class that tracks detailed statistics during gameplay."""
    
    def __init__(self, rounds: int, players: list[Player], verbose: bool = False):
        super().__init__(rounds, players, verbose)
        self.stats = {
            "ppo_final_score": 0,
            "opponent_final_score": 0,
            "ppo_points_per_round": [],
            "opponent_points_per_round": [],
            "ppo_busts": 0,
            "opponent_busts": 0,
            "banking_before_opponent": 0,
            "banking_after_opponent": 0,
            "banking_with_opponent": 0,
            "total_rolls": 0,
            "rolls_per_round": [],
        }
        self.ppo_id = 0  # PPO is always player 0
        self.opponent_id = 1  # Opponent is always player 1
    
    def play_round(self):
        """Play a round and track statistics."""
        score = self.first_rolls()
        players_in = [True for _ in range(len(self.players))]
        
        # Track points at start of round
        ppo_start_score = self.player_scores[self.ppo_id]
        opponent_start_score = self.player_scores[self.opponent_id]
        
        # Track banking decisions in this round
        ppo_has_banked = False
        opponent_has_banked = False
        
        k = 0  # Number of rolls in this round
        
        while True:
            state = {
                "current_score": score,
                "rounds_remaining": self.rounds - self.current_round,
                "player_scores": self.player_scores.copy(),
                "players_in": players_in.copy(),
            }
            
            # Collect all player decisions simultaneously
            decisions = {}
            for i, player in enumerate(self.players):
                if players_in[i]:
                    decisions[i] = player.decide_action(state)
                else:
                    decisions[i] = None
            
            # Track banking timing - check decisions before applying
            ppo_wants_to_bank = decisions[self.ppo_id] == "bank" and players_in[self.ppo_id]
            opponent_wants_to_bank = decisions[self.opponent_id] == "bank" and players_in[self.opponent_id]
            
            # Track if players bank this turn
            ppo_banks_this_turn = False
            opponent_banks_this_turn = False
            
            # Apply all decisions simultaneously
            for i, action in decisions.items():
                if action == "bank" and players_in[i]:
                    players_in[i] = False
                    self.player_scores[i] += score
                    if i == self.ppo_id:
                        ppo_banks_this_turn = True
                        ppo_has_banked = True
                    elif i == self.opponent_id:
                        opponent_banks_this_turn = True
                        opponent_has_banked = True
            
            # Track banking timing after applying decisions
            if ppo_banks_this_turn and opponent_banks_this_turn:
                # Both banked in the same turn
                self.stats["banking_with_opponent"] += 1
            elif ppo_banks_this_turn and not opponent_banks_this_turn:
                # Only PPO banked this turn
                if opponent_has_banked:
                    # Opponent already banked earlier in this round
                    self.stats["banking_after_opponent"] += 1
                else:
                    # Opponent still in, PPO banks first
                    self.stats["banking_before_opponent"] += 1
            elif opponent_banks_this_turn and not ppo_banks_this_turn:
                # Only opponent banked this turn (PPO still in)
                # Will count as "after" when PPO banks later
                pass
            
            if all(not player_in for player_in in players_in):
                break
            
            roll = self.roll()
            k += 1
            self.stats["total_rolls"] += 1
            
            if sum(roll) == 7:
                # Bust occurred - check who was still in when bust happened
                if players_in[self.ppo_id]:
                    self.stats["ppo_busts"] += 1
                if players_in[self.opponent_id]:
                    self.stats["opponent_busts"] += 1
                break
            elif roll[0] == roll[1]:
                score *= 2
            else:
                score += sum(roll)
        
        # Track points gained this round
        ppo_points_gained = self.player_scores[self.ppo_id] - ppo_start_score
        opponent_points_gained = self.player_scores[self.opponent_id] - opponent_start_score
        
        self.stats["ppo_points_per_round"].append(ppo_points_gained)
        self.stats["opponent_points_per_round"].append(opponent_points_gained)
        self.stats["rolls_per_round"].append(k)
        
        # Update player score history
        for i in range(len(self.players)):
            self.player_score_history[i].append(self.player_scores[i])
        
        return k, score
    
    def play_game(self):
        """Play a full game and collect final statistics."""
        super().play_game()
        
        # Store final scores
        self.stats["ppo_final_score"] = self.player_scores[self.ppo_id]
        self.stats["opponent_final_score"] = self.player_scores[self.opponent_id]
        
        return self.stats

# ============================================================================
# RL MODEL PLAYER WRAPPER
# ============================================================================

class RLModelPlayer(Player):
    """A Player wrapper that uses an RL model to make decisions."""
    
    def __init__(self, model, name: str = None, n_players: int = 2):
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

def run_game_with_stats(ppo_player: RLModelPlayer, opponent: Player, seed=None) -> Dict[str, Any]:
    """
    Run a single game and collect statistics.
    
    Returns:
        Dictionary with all game statistics
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create fresh copies of players
    fresh_ppo = copy.deepcopy(ppo_player)
    fresh_opponent = copy.deepcopy(opponent)
    
    # Create game with statistics tracking
    game = StatisticsBank(rounds=ROUNDS, players=[fresh_ppo, fresh_opponent], verbose=False)
    game.play_game()
    
    # Get statistics
    stats = game.stats.copy()
    
    # Calculate additional statistics
    stats["ppo_avg_points_per_round"] = np.mean(stats["ppo_points_per_round"]) if stats["ppo_points_per_round"] else 0.0
    stats["opponent_avg_points_per_round"] = np.mean(stats["opponent_points_per_round"]) if stats["opponent_points_per_round"] else 0.0
    stats["ppo_bust_percentage"] = stats["ppo_busts"] / ROUNDS if ROUNDS > 0 else 0.0
    stats["opponent_bust_percentage"] = stats["opponent_busts"] / ROUNDS if ROUNDS > 0 else 0.0
    stats["avg_rolls_per_round"] = np.mean(stats["rolls_per_round"]) if stats["rolls_per_round"] else 0.0
    
    return stats

def calculate_statistics(data_list: List[float]) -> Dict[str, float]:
    """Calculate mean, max, min, and variance for a list of values."""
    if not data_list:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "variance": 0.0}
    
    arr = np.array(data_list)
    return {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "variance": float(np.var(arr)),
    }

def collect_all_statistics(models: Dict[str, Any], opponents: Dict[str, Player]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Collect statistics for all model-opponent combinations.
    
    Returns:
        Nested dictionary: results[model_name][opponent_name] = statistics
    """
    all_results = {}
    
    for model_name, model_info in models.items():
        print(f"\n{'='*80}")
        print(f"Testing model: {model_name}")
        print(f"{'='*80}")
        
        model = model_info["model"]
        ppo_player = RLModelPlayer(model=model, name=model_name, n_players=2)
        
        all_results[model_name] = {}
        
        for opponent_name, opponent in opponents.items():
            print(f"  Testing against {opponent_name}...", end=" ", flush=True)
            
            # Collect statistics for 100 games
            game_stats = []
            for game_num in range(N_GAMES):
                stats = run_game_with_stats(ppo_player, opponent, seed=None)
                game_stats.append(stats)
            
            # Aggregate statistics
            aggregated = {
                "score_per_game": calculate_statistics([s["ppo_final_score"] for s in game_stats]),
                "points_per_round": calculate_statistics([s["ppo_avg_points_per_round"] for s in game_stats]),
                "ppo_bust_percentage": calculate_statistics([s["ppo_bust_percentage"] for s in game_stats]),
                "opponent_bust_percentage": calculate_statistics([s["opponent_bust_percentage"] for s in game_stats]),
                "banking_before_opponent": calculate_statistics([s["banking_before_opponent"] for s in game_stats]),
                "banking_after_opponent": calculate_statistics([s["banking_after_opponent"] for s in game_stats]),
                "banking_with_opponent": calculate_statistics([s["banking_with_opponent"] for s in game_stats]),
                "total_rolls": calculate_statistics([s["total_rolls"] for s in game_stats]),
                "avg_rolls_per_round": calculate_statistics([s["avg_rolls_per_round"] for s in game_stats]),
            }
            
            all_results[model_name][opponent_name] = aggregated
            print("Done")
    
    return all_results

def format_statistics_output(results: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """Format the statistics results as a structured text file."""
    output_lines = []
    
    # Header
    output_lines.append("=" * 100)
    output_lines.append("PPO TOURNAMENT STATISTICS")
    output_lines.append("=" * 100)
    output_lines.append("")
    output_lines.append(f"Configuration:")
    output_lines.append(f"  Games per matchup: {N_GAMES}")
    output_lines.append(f"  Rounds per game: {ROUNDS}")
    output_lines.append(f"  Total matchups: {len(results)} models × {len(list(results.values())[0])} opponents = {len(results) * len(list(results.values())[0])}")
    output_lines.append("")
    output_lines.append("=" * 100)
    output_lines.append("")
    
    # Statistics for each model
    for model_name, opponent_results in results.items():
        output_lines.append("")
        output_lines.append("=" * 100)
        output_lines.append(f"MODEL: {model_name.upper()}")
        output_lines.append("=" * 100)
        output_lines.append("")
        
        for opponent_name, stats in opponent_results.items():
            output_lines.append(f"  Opponent: {opponent_name.replace('_', ' ').title()}")
            output_lines.append(f"  {'-' * 96}")
            
            # Score per game
            s = stats["score_per_game"]
            output_lines.append(f"    Score per game:")
            output_lines.append(f"      Mean:    {s['mean']:>10.2f}")
            output_lines.append(f"      Max:     {s['max']:>10.2f}")
            output_lines.append(f"      Min:     {s['min']:>10.2f}")
            output_lines.append(f"      Variance: {s['variance']:>9.2f}")
            output_lines.append("")
            
            # Points per round
            s = stats["points_per_round"]
            output_lines.append(f"    Points per round (average):")
            output_lines.append(f"      Mean:    {s['mean']:>10.2f}")
            output_lines.append(f"      Max:     {s['max']:>10.2f}")
            output_lines.append(f"      Min:     {s['min']:>10.2f}")
            output_lines.append(f"      Variance: {s['variance']:>9.2f}")
            output_lines.append("")
            
            # Bust percentages
            s_ppo = stats["ppo_bust_percentage"]
            s_opp = stats["opponent_bust_percentage"]
            output_lines.append(f"    Bust percentage (PPO):")
            output_lines.append(f"      Mean:    {s_ppo['mean']:>10.4f} ({s_ppo['mean']*100:.2f}%)")
            output_lines.append(f"      Max:     {s_ppo['max']:>10.4f} ({s_ppo['max']*100:.2f}%)")
            output_lines.append(f"      Min:     {s_ppo['min']:>10.4f} ({s_ppo['min']*100:.2f}%)")
            output_lines.append(f"      Variance: {s_ppo['variance']:>9.6f}")
            output_lines.append("")
            output_lines.append(f"    Bust percentage (Opponent):")
            output_lines.append(f"      Mean:    {s_opp['mean']:>10.4f} ({s_opp['mean']*100:.2f}%)")
            output_lines.append(f"      Max:     {s_opp['max']:>10.4f} ({s_opp['max']*100:.2f}%)")
            output_lines.append(f"      Min:     {s_opp['min']:>10.4f} ({s_opp['min']*100:.2f}%)")
            output_lines.append(f"      Variance: {s_opp['variance']:>9.6f}")
            output_lines.append("")
            
            # Banking timing
            s_before = stats["banking_before_opponent"]
            s_after = stats["banking_after_opponent"]
            s_with = stats["banking_with_opponent"]
            output_lines.append(f"    Banking timing (counts per game):")
            output_lines.append(f"      Before opponent:")
            output_lines.append(f"        Mean:    {s_before['mean']:>10.2f}")
            output_lines.append(f"        Max:     {s_before['max']:>10.2f}")
            output_lines.append(f"        Min:     {s_before['min']:>10.2f}")
            output_lines.append(f"        Variance: {s_before['variance']:>9.2f}")
            output_lines.append(f"      After opponent:")
            output_lines.append(f"        Mean:    {s_after['mean']:>10.2f}")
            output_lines.append(f"        Max:     {s_after['max']:>10.2f}")
            output_lines.append(f"        Min:     {s_after['min']:>10.2f}")
            output_lines.append(f"        Variance: {s_after['variance']:>9.2f}")
            output_lines.append(f"      With opponent:")
            output_lines.append(f"        Mean:    {s_with['mean']:>10.2f}")
            output_lines.append(f"        Max:     {s_with['max']:>10.2f}")
            output_lines.append(f"        Min:     {s_with['min']:>10.2f}")
            output_lines.append(f"        Variance: {s_with['variance']:>9.2f}")
            output_lines.append("")
            
            # Rolls
            s_total = stats["total_rolls"]
            s_avg = stats["avg_rolls_per_round"]
            output_lines.append(f"    Total rolls per game:")
            output_lines.append(f"      Mean:    {s_total['mean']:>10.2f}")
            output_lines.append(f"      Max:     {s_total['max']:>10.2f}")
            output_lines.append(f"      Min:     {s_total['min']:>10.2f}")
            output_lines.append(f"      Variance: {s_total['variance']:>9.2f}")
            output_lines.append("")
            output_lines.append(f"    Average rolls per round:")
            output_lines.append(f"      Mean:    {s_avg['mean']:>10.2f}")
            output_lines.append(f"      Max:     {s_avg['max']:>10.2f}")
            output_lines.append(f"      Min:     {s_avg['min']:>10.2f}")
            output_lines.append(f"      Variance: {s_avg['variance']:>9.2f}")
            output_lines.append("")
    
    return "\n".join(output_lines)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function to run the statistics collection."""
    print("=" * 80)
    print("PPO Statistics Collection")
    print("=" * 80)
    print(f"Testing {len(MULTI_MODEL_PATHS)} multi PPO models and {len(TROLL_MODEL_PATHS)} troll PPO models")
    print(f"Against {len(OPPONENTS)} opponent types")
    print(f"Running {N_GAMES} games per matchup")
    print(f"Total games: {(len(MULTI_MODEL_PATHS) + len(TROLL_MODEL_PATHS)) * len(OPPONENTS) * N_GAMES}")
    print()
    
    # Combine all models
    all_models = {**MULTI_MODEL_PATHS, **TROLL_MODEL_PATHS}
    
    # Check all models exist
    print("Checking model files...")
    missing_models = []
    for model_name, model_info in all_models.items():
        if not model_info["path"].exists():
            missing_models.append((model_name, model_info["path"]))
            print(f"  ✗ {model_name}: {model_info['path']} - NOT FOUND")
        else:
            print(f"  ✓ {model_name}: {model_info['path']}")
    
    if missing_models:
        print(f"\nError: {len(missing_models)} model(s) not found. Exiting.")
        return
    
    print()
    
    # Load all models
    print("Loading models...")
    loaded_models = {}
    for model_name, model_info in all_models.items():
        print(f"  Loading {model_name}...", end=" ", flush=True)
        try:
            model = load_model(model_info["path"], model_info["folder"])
            loaded_models[model_name] = {
                "model": model,
                "folder": model_info["folder"]
            }
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print()
    
    # Collect statistics
    print("Collecting statistics...")
    results = collect_all_statistics(loaded_models, OPPONENTS)
    
    # Format and save output
    print("\nFormatting output...")
    output_text = format_statistics_output(results)
    
    # Save to file
    output_file = project_root / "player_arena" / "ppo_statistics_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(output_text)
    
    print(f"\nStatistics saved to: {output_file}")
    print("\nStatistics collection complete!")

if __name__ == "__main__":
    main()

