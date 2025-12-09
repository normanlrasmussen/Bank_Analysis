"""
RL Tournament: All RL models play against each other

This script loads all trained RL models (PPO, PPO_multi, A2C, DQN) and has them
play against each other in 4-player games. Results are visualized
as win percentages.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import copy
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
    from stable_baselines3 import PPO, A2C, DQN
    from rl_og.bank_gym import BankEnv
    from rl_og.players import Player
except ModuleNotFoundError as e:
    if 'game_theory' in str(e):
        # Ensure project_root is definitely in the path
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        # Try importing again
        from stable_baselines3 import PPO, A2C, DQN
        from rl_og.bank_gym import BankEnv
        from rl_og.players import Player
    else:
        raise


class RLModelPlayer(Player):
    """
    Wrapper class that makes an RL model act like a Player.
    This allows RL models to be used as opponents in BankEnv.
    
    Note: Models were trained with different numbers of opponents, but we use them
    in 4-player games (10-dim obs). We convert the 4-player state to match
    the observation format each model expects.
    """
    def __init__(self, model, env_template):
        """
        Args:
            model: The loaded RL model (PPO, A2C, or DQN)
            env_template: A template BankEnv for 4-player games (to get n_players, rounds, etc.)
        """
        super().__init__()
        self.model = model
        self.env_template = env_template
        self.current_obs = None
        # Get the expected observation space size from the model
        self.obs_dim = model.observation_space.shape[0]
    
    def decide_action(self, state):
        """
        Convert the game state to an observation and get action from model.
        
        Args:
            state: Dictionary with game state information
        
        Returns:
            "roll" or "bank"
        """
        # Convert state to observation format
        obs = self._state_to_observation(state)
        
        # Get action from model (deterministic for evaluation)
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Convert action (0=roll, 1=bank) to string
        return "bank" if action == 1 else "roll"
    
    def _state_to_observation(self, state):
        """
        Convert game state dictionary to observation array.
        
        Handles different observation space sizes:
        - 6-dim: trained with 1 opponent [score, rounds, p0, p1, in0, in1]
        - 12-dim: trained with 4 opponents [score, rounds, p0, p1, p2, p3, p4, in0, in1, in2, in3, in4]
        """
        # Extract values from state
        current_score = state.get("current_score", 0)
        rounds_remaining = state.get("rounds_remaining", 0)
        player_scores = state.get("player_scores", [])
        players_in = state.get("players_in", [])
        
        rounds = self.env_template.rounds
        score_scale = 200.0 * rounds if rounds > 0 else 200.0
        
        # Create observation with the size the model expects
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        # Normalize current_score (same as BankEnv)
        obs[0] = np.tanh(float(current_score) / 200.0)
        
        # Normalize rounds_remaining
        obs[1] = float(rounds_remaining) / float(max(rounds, 1))
        
        if self.obs_dim == 6:
            # 6-dim: 1 opponent model - convert to 2-player format
            # Use agent's perspective: agent vs best opponent
            if len(player_scores) >= 4:
                # 4-player game: use agent vs best of 3 opponents
                agent_score = player_scores[0]
                opponent_scores = player_scores[1:4]
                max_opponent_score = max(opponent_scores) if opponent_scores else 0
                
                obs[2] = np.tanh(float(agent_score) / score_scale)  # "my" score
                obs[3] = np.tanh(float(max_opponent_score) / score_scale)  # "opponent" score
                
                agent_in = players_in[0] if len(players_in) > 0 else False
                any_opponent_in = any(players_in[1:4]) if len(players_in) >= 4 else False
                obs[4] = float(agent_in)
                obs[5] = float(any_opponent_in)
            elif len(player_scores) >= 3:
                # 3-player game: use agent vs best of 2 opponents
                agent_score = player_scores[0]
                opponent_scores = player_scores[1:3]
                max_opponent_score = max(opponent_scores) if opponent_scores else 0
                
                obs[2] = np.tanh(float(agent_score) / score_scale)
                obs[3] = np.tanh(float(max_opponent_score) / score_scale)
                
                agent_in = players_in[0] if len(players_in) > 0 else False
                any_opponent_in = any(players_in[1:3]) if len(players_in) >= 3 else False
                obs[4] = float(agent_in)
                obs[5] = float(any_opponent_in)
            else:
                # Fallback for 2-player case
                for i in range(min(2, len(player_scores))):
                    obs[2 + i] = np.tanh(float(player_scores[i]) / score_scale)
                for i in range(min(2, len(players_in))):
                    obs[4 + i] = float(players_in[i])
        
        elif self.obs_dim == 12:
            # 12-dim: 4 opponent model - convert to 5-player format
            # Format: [score, rounds, p0, p1, p2, p3, p4, in0, in1, in2, in3, in4]
            if len(player_scores) >= 4:
                # 4-player game: [agent, opp1, opp2, opp3] -> map to [agent, opp1, opp2, opp3, 0]
                for i in range(min(4, len(player_scores))):
                    obs[2 + i] = np.tanh(float(player_scores[i]) / score_scale)
                # obs[6] remains 0 (no p4)
                
                for i in range(min(4, len(players_in))):
                    obs[7 + i] = float(players_in[i])
                # obs[11] remains 0 (no in4)
            else:
                # 3-player game: [agent, opp1, opp2] -> map to [agent, opp1, opp2, 0, 0]
                for i in range(min(3, len(player_scores))):
                    obs[2 + i] = np.tanh(float(player_scores[i]) / score_scale)
                # obs[5], obs[6] remain 0
                
                for i in range(min(3, len(players_in))):
                    obs[7 + i] = float(players_in[i])
                # obs[10], obs[11] remain 0
        
        else:
            # Fallback: try to fill what we can
            n_score_slots = (self.obs_dim - 2) // 2  # Half for scores, half for players_in
            for i in range(min(n_score_slots, len(player_scores))):
                obs[2 + i] = np.tanh(float(player_scores[i]) / score_scale)
            for i in range(min(n_score_slots, len(players_in))):
                obs[2 + n_score_slots + i] = float(players_in[i])
        
        return obs


def convert_obs(obs_source, source_dim, target_dim):
    """
    Convert observation from source dimension to target dimension.
    
    Source formats:
    - 8-dim (3 players): [score, rounds, p0, p1, p2, in0, in1, in2]
    - 10-dim (4 players): [score, rounds, p0, p1, p2, p3, in0, in1, in2, in3]
    
    Target formats:
    - 6-dim (2 players): [score, rounds, p0, max(opponents), in0, any(opponents_in)]
    - 12-dim (5 players): [score, rounds, p0, p1, p2, p3, p4, in0, in1, in2, in3, in4]
    """
    obs = np.zeros(target_dim, dtype=np.float32)
    
    # Copy score and rounds (first 2 elements)
    obs[0] = obs_source[0]
    obs[1] = obs_source[1]
    
    if target_dim == 6:
        # 6-dim: 2-player format - use agent vs best opponent
        obs[2] = obs_source[2]  # Agent score (p0)
        if source_dim == 8:  # 3 players
            obs[3] = max(obs_source[3], obs_source[4]) if len(obs_source) > 4 else obs_source[3]
            obs[4] = obs_source[5]  # Agent in
            obs[5] = max(obs_source[6], obs_source[7]) if len(obs_source) > 7 else obs_source[6]
        elif source_dim == 10:  # 4 players
            obs[3] = max(obs_source[3], obs_source[4], obs_source[5]) if len(obs_source) > 5 else max(obs_source[3], obs_source[4])
            obs[4] = obs_source[6]  # Agent in
            obs[5] = max(obs_source[7], obs_source[8], obs_source[9]) if len(obs_source) > 9 else max(obs_source[7], obs_source[8])
    elif target_dim == 12:
        # 12-dim: 5-player format (pad with zeros)
        if source_dim == 8:  # 3 players -> pad to 5
            obs[2] = obs_source[2]  # p0
            obs[3] = obs_source[3]  # p1
            obs[4] = obs_source[4]  # p2
            # obs[5], obs[6] remain 0
            obs[7] = obs_source[5]  # in0
            obs[8] = obs_source[6]  # in1
            obs[9] = obs_source[7]  # in2
            # obs[10], obs[11] remain 0
        elif source_dim == 10:  # 4 players -> pad to 5
            obs[2] = obs_source[2]  # p0
            obs[3] = obs_source[3]  # p1
            obs[4] = obs_source[4]  # p2
            obs[5] = obs_source[5]  # p3
            # obs[6] remains 0
            obs[7] = obs_source[6]  # in0
            obs[8] = obs_source[7]  # in1
            obs[9] = obs_source[8]  # in2
            obs[10] = obs_source[9]  # in3
            # obs[11] remains 0
    else:
        # Fallback: try to fill what we can
        n_score_slots = (target_dim - 2) // 2
        n_players_in_source = (source_dim - 2) // 2
        for i in range(min(n_players_in_source, n_score_slots)):
            obs[2 + i] = obs_source[2 + i] if len(obs_source) > 2 + i else 0
        for i in range(min(n_players_in_source, n_score_slots)):
            obs[2 + n_score_slots + i] = obs_source[2 + n_players_in_source + i] if len(obs_source) > 2 + n_players_in_source + i else 0
    
    return obs


def run_4player_game(model_a, model_b, model_c, model_d, env_template, seed=None):
    """
    Run a single 4-player game with all four RL models playing simultaneously.
    
    Since BankEnv only supports 1 RL agent at a time, we run one game where
    one model is the agent and the other three are opponents. All four models
    participate in the same game and we get their final scores.
    
    Args:
        model_a: First RL model
        model_b: Second RL model
        model_c: Third RL model
        model_d: Fourth RL model
        env_template: Template environment for creating the game
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with final scores: {'A': score_a, 'B': score_b, 'C': score_c, 'D': score_d}
        where scores are from a single game with all 4 models playing
    """
    # Use model A as the agent, B, C, and D as opponents
    # (We'll rotate which model is agent across games for fairness)
    opponent_b = RLModelPlayer(model_b, env_template)
    opponent_c = RLModelPlayer(model_c, env_template)
    opponent_d = RLModelPlayer(model_d, env_template)
    env = BankEnv(
        rounds=env_template.rounds,
        opponents=[opponent_b, opponent_c, opponent_d],
        max_round_length=env_template.max_round_length,
        verbose=False
    )
    # Don't call set_env() - it checks observation space compatibility
    # Instead, we'll convert observations manually based on model's expected size
    obs_10dim, info = env.reset(seed=seed)
    target_obs_dim = model_a.observation_space.shape[0]
    obs_converted = convert_obs(obs_10dim, 10, target_obs_dim)
    
    terminated = False
    truncated = False
    while not (terminated or truncated):
        # Use converted observation for model prediction
        action, _ = model_a.predict(obs_converted, deterministic=True)
        obs_10dim, reward, terminated, truncated, info = env.step(action)
        # Convert next observation to model's expected size
        obs_converted = convert_obs(obs_10dim, 10, target_obs_dim)
    
    # Get final scores: [agent_score, opponent1_score, opponent2_score, opponent3_score]
    final_scores = info.get('final_scores', info.get('player_scores', [0, 0, 0, 0]))
    
    # Map scores to models: A is agent (index 0), B is opponent1 (index 1), C is opponent2 (index 2), D is opponent3 (index 3)
    scores = {
        'A': final_scores[0] if len(final_scores) > 0 else 0,
        'B': final_scores[1] if len(final_scores) > 1 else 0,
        'C': final_scores[2] if len(final_scores) > 2 else 0,
        'D': final_scores[3] if len(final_scores) > 3 else 0
    }
    
    env.close()
    return scores


def main():
    """Main tournament function."""
    print("=" * 80)
    print("RL Tournament: All Models vs All Models")
    print("=" * 80)
    
    # Path to models - check both rl_og/RL_data and root RL_data
    rl_data_path = project_root / "rl_og" / "RL_data"
    root_rl_data_path = project_root / "RL_data"
    
    # Check both locations for each model file
    def find_model_file(filename):
        """Find model file in either rl_og/RL_data or root RL_data directory."""
        path1 = rl_data_path / filename
        path2 = root_rl_data_path / filename
        if path1.exists():
            return path1
        elif path2.exists():
            return path2
        return None
    
    ppo_old_model_path = find_model_file("rl_bank_model_ppo.zip")
    ppo_multi_model_path = find_model_file("rl_bank_model_ppo_multi.zip")
    a2c_model_path = find_model_file("rl_bank_model_a2c.zip")
    dqn_model_path = find_model_file("rl_bank_model_dqn.zip")
    
    # Create template environments
    # Most models were trained with 1 opponent, but PPO_multi was trained with 4 opponents
    from rl_og.players import ThersholdPlayer, ProbabilisticPlayer
    load_template_opponents = [ThersholdPlayer(threshold=100)]
    load_template_env = BankEnv(
        rounds=10,
        opponents=load_template_opponents,
        max_round_length=100,
        verbose=False
    )
    
    # Template for PPO_multi model (trained with 4 opponents)
    load_template_opponents_multi = [
        ThersholdPlayer(threshold=100),
        ThersholdPlayer(threshold=200),
        ThersholdPlayer(threshold=2000),
        ProbabilisticPlayer(probability=0.2),
    ]
    load_template_env_multi = BankEnv(
        rounds=10,
        opponents=load_template_opponents_multi,
        max_round_length=100,
        verbose=False
    )
    
    # Template for 4-player games (1 agent + 3 opponents)
    game_template_opponents = [ThersholdPlayer(threshold=100), ThersholdPlayer(threshold=100), ThersholdPlayer(threshold=100)]
    game_template_env = BankEnv(
        rounds=10,
        opponents=game_template_opponents,
        max_round_length=100,
        verbose=False
    )
    
    # Load all available models (suppress stable-baselines3 print statements)
    # PPO_multi was trained with 4 opponents, so needs matching environment for loading
    models = {}
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull):
            if ppo_old_model_path is not None and ppo_old_model_path.exists():
                models["PPO"] = PPO.load(str(ppo_old_model_path), env=load_template_env)
                print("✓ Loaded PPO (old) model")
            if ppo_multi_model_path is not None and ppo_multi_model_path.exists():
                models["PPO_multi"] = PPO.load(str(ppo_multi_model_path), env=load_template_env_multi)
                print("✓ Loaded PPO (multi) model")
            if a2c_model_path is not None and a2c_model_path.exists():
                models["A2C"] = A2C.load(str(a2c_model_path), env=load_template_env)
                print("✓ Loaded A2C model")
            if dqn_model_path is not None and dqn_model_path.exists():
                models["DQN"] = DQN.load(str(dqn_model_path), env=load_template_env)
                print("✓ Loaded DQN model")
    
    if len(models) != 4:
        print(f"Error: Need exactly 4 models for 4-player tournament, but found {len(models)}")
        return
    
    model_names = list(models.keys())
    model_a_name, model_b_name, model_c_name, model_d_name = model_names
    model_a = models[model_a_name]
    model_b = models[model_b_name]
    model_c = models[model_c_name]
    model_d = models[model_d_name]
    
    print(f"\nFound {len(models)} models: {', '.join(model_names)}")
    print(f"Running 1000 four-player games (all models play simultaneously)...")
    print()
    
    # Track wins for each model
    wins = {name: 0 for name in model_names}
    ties = {name: 0 for name in model_names}
    total_scores = {name: 0 for name in model_names}
    
    # Run 1000 games, rotating which model is the agent for fairness
    n_games = 1000
    for game in range(n_games):
        if (game + 1) % 100 == 0:
            print(f"Progress: {game + 1}/{n_games} games completed...", flush=True)
        
        # Rotate which model is the agent (player 0) for fairness
        # Game % 4 == 0: A is agent, B, C, D are opponents
        # Game % 4 == 1: B is agent, A, C, D are opponents
        # Game % 4 == 2: C is agent, A, B, D are opponents
        # Game % 4 == 3: D is agent, A, B, C are opponents
        rotation = game % 4
        
        if rotation == 0:
            # A is agent
            scores = run_4player_game(model_a, model_b, model_c, model_d, game_template_env, seed=None)
            score_map = {
                model_a_name: scores['A'],
                model_b_name: scores['B'],
                model_c_name: scores['C'],
                model_d_name: scores['D']
            }
        elif rotation == 1:
            # B is agent
            scores = run_4player_game(model_b, model_a, model_c, model_d, game_template_env, seed=None)
            score_map = {
                model_b_name: scores['A'],  # B is agent, so it's 'A' in the result
                model_a_name: scores['B'],  # A is opponent1, so it's 'B' in the result
                model_c_name: scores['C'],  # C is opponent2, so it's 'C' in the result
                model_d_name: scores['D']   # D is opponent3, so it's 'D' in the result
            }
        elif rotation == 2:
            # C is agent
            scores = run_4player_game(model_c, model_a, model_b, model_d, game_template_env, seed=None)
            score_map = {
                model_c_name: scores['A'],  # C is agent, so it's 'A' in the result
                model_a_name: scores['B'],  # A is opponent1, so it's 'B' in the result
                model_b_name: scores['C'],  # B is opponent2, so it's 'C' in the result
                model_d_name: scores['D']   # D is opponent3, so it's 'D' in the result
            }
        else:  # rotation == 3
            # D is agent
            scores = run_4player_game(model_d, model_a, model_b, model_c, game_template_env, seed=None)
            score_map = {
                model_d_name: scores['A'],  # D is agent, so it's 'A' in the result
                model_a_name: scores['B'],  # A is opponent1, so it's 'B' in the result
                model_b_name: scores['C'],  # B is opponent2, so it's 'C' in the result
                model_c_name: scores['D']   # C is opponent3, so it's 'D' in the result
            }
        
        # Determine winner(s)
        max_score = max(score_map.values())
        winners = [name for name, score in score_map.items() if score == max_score]
        
        # Update statistics
        for name in model_names:
            total_scores[name] += score_map[name]
            if len(winners) == 1 and name in winners:
                wins[name] += 1
            elif len(winners) > 1 and name in winners:
                ties[name] += 1
    
    load_template_env.close()
    if ppo_multi_model_path is not None:
        load_template_env_multi.close()
    game_template_env.close()
    
    # Calculate win percentages
    win_percentages = {}
    avg_scores = {}
    for name in model_names:
        win_percentages[name] = (wins[name] + 0.5 * ties[name]) / n_games
        avg_scores[name] = total_scores[name] / n_games
    
    # Print summary
    print("\n" + "=" * 80)
    print("Tournament Results Summary (4-Player Games)")
    print("=" * 80)
    print(f"\nTotal games run: {n_games}")
    print("\nWin Statistics:")
    for name in model_names:
        losses = n_games - wins[name] - ties[name]
        print(f"  {name:12s}: {wins[name]:4d} wins, {ties[name]:4d} ties, "
              f"{losses:4d} losses")
    
    print("\nWin Percentages:")
    for name in model_names:
        print(f"  {name:12s}: {win_percentages[name]:.3f} ({win_percentages[name]*100:.1f}%)")
    
    print("\nAverage Scores:")
    for name in model_names:
        print(f"  {name:12s}: {avg_scores[name]:.2f}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of win percentages
    ax1 = axes[0]
    win_rates = [win_percentages[name] for name in model_names]
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown'][:len(model_names)]
    bars1 = ax1.bar(model_names, win_rates, color=colors)
    ax1.set_ylabel('Win Percentage')
    ax1.set_title('Win Percentages\n(4-Player Games)')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, rate in zip(bars1, win_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Bar chart of average scores
    ax2 = axes[1]
    avg_scores_list = [avg_scores[name] for name in model_names]
    bars2 = ax2.bar(model_names, avg_scores_list, color=colors)
    ax2.set_ylabel('Average Score')
    ax2.set_title('Average Scores\n(4-Player Games)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars2, avg_scores_list):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\nTournament complete!")


if __name__ == "__main__":
    main()

