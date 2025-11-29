import numpy as np
import copy
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from players import Player

class RLPlayer(Player):
            def __init__(self, env):
                super().__init__()
                self.env = env
            
            def decide_action(self, state):
                if self.env.agent_banked:
                    return "bank"
                return "roll"


class BankEnv(gym.Env):
    """
    Gymnasium environment wrapper for the Bank push-your-luck dice game.
    
    This environment allows training a single RL agent against other players.
    The RL agent is always player 0, while other players can be instances
    of the Player class from player.py.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        rounds: int = 10,
        opponents: Optional[list[Player]] = None,
        max_round_length: int = 1000,
        verbose: bool = False,
        render_mode: Optional[str] = None,
        reward_mode: str = "round",  # "round" or "sparse"
    ):
        """
        Initialize the Bank Gymnasium environment.
        
        Args:
            rounds: Number of rounds to play
            opponents: List of opponent Player instances (can be None for 1-player mode)
            max_round_length: Maximum steps per round to prevent infinite loops
            verbose: Whether to print game information
            render_mode: Rendering mode ("human" or "rgb_array")
            reward_mode: Reward calculation mode - "round" for incremental round-based rewards,
                        "sparse" for end-of-game only rewards
        """
        super().__init__()
        
        self.rounds = rounds
        self.max_round_length = max_round_length
        self.verbose = verbose
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        
        # Set up opponents (empty list if None)
        if opponents is None:
            opponents = []
        self.opponents = opponents
        self.n_opponents = len(opponents)
        self.n_players = 1 + self.n_opponents
        
        # Action space: 0 = roll, 1 = bank
        self.action_space = spaces.Discrete(2)
        
        # Observation space: current_score, rounds_remaining, player_scores, players_in
        obs_dim = 1 + 1 + self.n_players + self.n_players
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize game state
        self.current_round = 0
        self.round_step = 0
        self.current_state = None
        self.agent_banked = False
        self._player_scores = np.zeros(self.n_players, dtype=np.float32)
        self._players_in = np.ones(self.n_players, dtype=bool)
        self._obs_buffer = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._prev_score_diff = 0.0  # Track previous score difference for round-based rewards
        
    def _get_observation(self) -> np.ndarray:
        """
        Convert the game state dictionary to a numpy array observation.
        Observations are normalized to help with learning stability.
        
        Returns:
            Observation array: [current_score, rounds_remaining, player_0_score, ..., player_n_score, 
                               player_0_in, ..., player_n_in]
        """
        if self.current_state is None:
            # Return zero observation if no state available
            self._obs_buffer.fill(0.0)
            return self._obs_buffer
        
        obs = self._obs_buffer
        obs.fill(0.0)
        
        # Normalize Scores
        current_score_scale = 5000.0
        obs[0] = np.tanh(float(self.current_state["current_score"]) / current_score_scale)
        obs[1] = float(self.current_state["rounds_remaining"]) / float(max(self.rounds, 1))
        score_scale = 1000.0 * self.rounds if self.rounds > 0 else 1000.0
        player_scores = self.current_state["player_scores"]
        for i in range(self.n_players):
            obs[2 + i] = np.tanh(float(player_scores[i]) / score_scale)
        
        players_in = self.current_state["players_in"]
        for i in range(self.n_players):
            obs[2 + self.n_players + i] = float(players_in[i])
        
        return obs
    
    def _end_round(self, player_scores: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Handle round ending and transition to next round or game end."""
        terminated = False
        truncated = False
        
        # Update current_state before computing reward (needed for round-based rewards)
        if self.reward_mode == "round":
            self.current_state = {
                "current_score": 0,
                "rounds_remaining": self.rounds - self.current_round,
                "player_scores": player_scores,
                "players_in": self._players_in,
            }
        
        reward = self._calculate_reward()
        
        self.current_round += 1
        self.round_step = 0
        self.agent_banked = False
        
        if self.current_round > self.rounds:
            final_scores = player_scores.copy()
            reward += self._calculate_final_reward(player_scores)
            self.current_state = None
            terminated = True
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"game_ended": True, "final_scores": final_scores.tolist()}
            return observation, reward, terminated, truncated, info
        
        score = self.first_rolls()
        self._players_in.fill(True)
        self.current_state = {
            "current_score": score,
            "rounds_remaining": self.rounds - self.current_round + 1,
            "player_scores": player_scores,
            "players_in": self._players_in,
        }
        
        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, terminated, truncated, info
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the current state."""
        if self.current_state is None:
            return {}
        player_scores = self.current_state["player_scores"]
        players_in = self.current_state["players_in"]
        return {
            "current_score": self.current_state["current_score"],
            "rounds_remaining": self.current_state["rounds_remaining"],
            "player_scores": player_scores.tolist() if isinstance(player_scores, np.ndarray) else list(player_scores),
            "players_in": players_in.tolist() if isinstance(players_in, np.ndarray) else list(players_in),
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Information dictionary
        """
        super().reset(seed=seed)
        
        
        fresh_opponents = [copy.deepcopy(opp) for opp in self.opponents]
        rl_player = RLPlayer(self)
        self.all_players = [rl_player] + fresh_opponents
        
        for i, player in enumerate(self.all_players):
            player.set_player_id(i)
        
        self._player_scores.fill(0.0)
        self._prev_score_diff = 0.0  # Reset previous score difference for round-based rewards
        
        self.current_round = 1
        self.round_step = 0
        self.agent_banked = False
        score = self.first_rolls()
        
        self._players_in.fill(True)
        self.current_state = {
            "current_score": score,
            "rounds_remaining": self.rounds - self.current_round + 1,
            "player_scores": self._player_scores,
            "players_in": self._players_in,
        }

        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0 = roll, 1 = bank
            
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        
        # Handle case where environment has already ended
        if self.current_state is None:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"game_ended": True}
            return observation, 0.0, True, False, info
        
        score = self.current_state["current_score"]
        players_in = self.current_state["players_in"]
        player_scores = self.current_state["player_scores"]
        
        decisions = {}
        
        if players_in[0]:
            decisions[0] = "bank" if action == 1 else "roll"
        else:
            decisions[0] = None
        
        for i in range(1, self.n_players):
            if players_in[i]:
                state_for_opponent = {
                    "current_score": score,
                    "rounds_remaining": self.current_state["rounds_remaining"],
                    "player_scores": player_scores.tolist() if isinstance(player_scores, np.ndarray) else list(player_scores),
                    "players_in": players_in.tolist() if isinstance(players_in, np.ndarray) else list(players_in),
                }
                if not hasattr(self.all_players[i], 'player_id') or self.all_players[i].player_id != i:
                    self.all_players[i].player_id = i
                decisions[i] = self.all_players[i].decide_action(state_for_opponent)
            else:
                decisions[i] = None
        
        for i in range(self.n_players):
            if decisions[i] == "bank" and players_in[i]:
                players_in[i] = False
                player_scores[i] += score
                if i == 0:
                    self.agent_banked = True
        
        if not players_in.any():
            return self._end_round(player_scores)
        
        roll = self.roll()
        roll_sum = roll[0] + roll[1]
        
        if roll_sum == 7:
            return self._end_round(player_scores)
        
        if roll[0] == roll[1]:
            score *= 2
        else:
            score += roll_sum
        
        self.round_step += 1
        truncated = self.round_step >= self.max_round_length
        self.current_state["current_score"] = score
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, 0.0, False, truncated, info
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on the selected reward mode.
        
        - "round": Incremental reward based on score difference at end of each round
        - "sparse": Returns 0.0 (rewards only at game end)
        """
        if self.reward_mode == "round":
            return self._calculate_round_reward()
        else:  # sparse mode
            return 0.0
    
    def _calculate_round_reward(self) -> float:
        """
        Reward given only at the *end of a round* (round-based mode).
        Reward = (agent_total_score - max_opponent_total_score) - previous_score_diff.
        This gives the incremental change in score difference for this round.
        Rewards are normalized to be on a similar scale to sparse rewards for consistency.
        """
        if self.current_state is None:
            return 0.0

        scores = self.current_state["player_scores"]
        agent_score = scores[0]
        if len(scores) > 1:
            max_opp = np.max(scores[1:])
        else:
            max_opp = 0.0

        # Normalize score difference to keep rewards in a reasonable range
        # Using a fixed scale factor (1000.0) to ensure rewards are meaningful
        # This keeps incremental rewards per round in a similar magnitude to sparse rewards
        score_scale = 1000.0
        current_score_diff = float(agent_score - max_opp) / score_scale
        reward = current_score_diff - self._prev_score_diff
        self._prev_score_diff = current_score_diff
        return reward
    
    def _calculate_final_reward(self, player_scores: np.ndarray) -> float:
        """
        Calculate final reward at the end of the game.
        
        - "round": Returns 0.0 (main reward comes from per-round rewards)
        - "sparse": Returns 1.0 for win, 0.0 for tie, -1.0 for loss
        """
        if self.reward_mode == "sparse":
            if len(player_scores) == 1:
                return 0.0
            if player_scores[0] > max(player_scores[1:]):
                return 1.0
            elif player_scores[0] == max(player_scores[1:]):
                return 0.0
            else:
                return -1.0
        else:  # round mode
            # The per-round rewards should already reflect performance
            # This is just a small bonus/penalty for final outcome
            # The main reward comes from _calculate_round_reward above
            return 0.0
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.current_state:
                print(f"Round: {self.current_round}/{self.rounds}")
                print(f"Current Score: {self.current_state['current_score']}")
                print(f"Player Scores: {self.current_state['player_scores']}")
                print(f"Players In: {self.current_state['players_in']}")
                print("-" * 40)
    
    def close(self):
        """Clean up resources."""
        pass

    def roll(self):
        """Roll two dice and return the result."""
        return np.random.randint(1, 7), np.random.randint(1, 7)
    
    def first_rolls(self):
        """Roll 3 dice and return the sum, treating 7 as 70."""
        total = 0
        for _ in range(3):
            roll_sum = sum(self.roll())
            total += 70 if roll_sum == 7 else roll_sum
        return total