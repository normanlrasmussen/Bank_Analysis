import numpy as np
import copy
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from game_theory.bank import Bank
from players import Player


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
        max_round_length: int = 100,
        verbose: bool = False,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the Bank Gymnasium environment.
        
        Args:
            rounds: Number of rounds to play
            opponents: List of opponent Player instances (can be None for 1-player mode)
            max_round_length: Maximum steps per round to prevent infinite loops
            verbose: Whether to print game information
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()
        
        self.rounds = rounds
        self.max_round_length = max_round_length
        self.verbose = verbose
        self.render_mode = render_mode
        
        # Set up opponents (empty list if None)
        if opponents is None:
            opponents = []
        self.opponents = opponents
        self.n_opponents = len(opponents)
        self.n_players = 1 + self.n_opponents
        
        # Action space: 0 = roll, 1 = bank
        self.action_space = spaces.Discrete(2)
        
        # Observation space: current_score, rounds_remaining, player_scores, players_in
        # We'll use a Box space for the observation vector
        obs_dim = 1 + 1 + self.n_players + self.n_players  # score, rounds, scores array, players_in array
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize game state
        self.bank = None
        self.current_round = 0
        self.round_step = 0
        self.current_state = None
        self.agent_banked = False
        
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
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Normalize current_score (typical range: 0-500, normalize to 0-1)
        # Use tanh normalization to handle outliers gracefully
        obs[0] = np.tanh(float(self.current_state["current_score"]) / 200.0)
        
        # Normalize rounds_remaining (range: 0 to self.rounds)
        obs[1] = float(self.current_state["rounds_remaining"]) / float(max(self.rounds, 1))
        
        # Normalize player scores - use adaptive scaling based on rounds
        # For single round: ~0-200, for 10 rounds: ~0-2000
        score_scale = 200.0 * self.rounds if self.rounds > 0 else 200.0
        for i in range(self.n_players):
            obs[2 + i] = np.tanh(float(self.current_state["player_scores"][i]) / score_scale)
        
        # Players still in the round (already 0 or 1, no normalization needed)
        for i in range(self.n_players):
            obs[2 + self.n_players + i] = float(self.current_state["players_in"][i])
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the current state."""
        if self.current_state is None:
            return {}
        return {
            "current_score": self.current_state["current_score"],
            "rounds_remaining": self.current_state["rounds_remaining"],
            "player_scores": self.current_state["player_scores"].copy(),
            "players_in": self.current_state["players_in"].copy(),
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
        
        # Create a dummy RL player that we'll control via actions
        from players import Player as BasePlayer
        class RLPlayer(BasePlayer):
            def __init__(self, env):
                super().__init__()
                self.env = env
                self.action = None
            
            def decide_action(self, state):
                # This will be set by the step() method
                # Return "roll" by default, but step() will override
                if self.env.agent_banked:
                    return "bank"
                return self.env.agent_action
        
        # Create all players: RL agent first, then opponents
        # Need to deep copy opponents to avoid state sharing
        fresh_opponents = [copy.deepcopy(opp) for opp in self.opponents]
        rl_player = RLPlayer(self)
        all_players = [rl_player] + fresh_opponents
        
        # Initialize the Bank game
        self.bank = Bank(self.rounds, all_players, verbose=self.verbose)
        self.bank.current_round = 1
        self.bank.player_scores = [0 for _ in range(self.n_players)]
        self.bank.player_score_history = [[] for _ in range(self.n_players)]
        
        # Ensure all players have player_ids set
        for i, player in enumerate(self.bank.players):
            player.set_player_id(i)
        
        # Start the first round
        self.current_round = 1
        self.round_step = 0
        self.agent_banked = False
        score = self.bank.first_rolls()
        
        players_in = [True for _ in range(self.n_players)]
        self.current_state = {
            "current_score": score,
            "rounds_remaining": self.rounds - self.current_round + 1,
            "player_scores": self.bank.player_scores.copy(),
            "players_in": players_in.copy(),
        }
        
        # Store reference to RL player in bank
        self.bank.rl_player = rl_player
        
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
        if self.bank is None:
            raise RuntimeError("Environment must be reset before calling step()")
        
        if self.current_state is None:
            raise RuntimeError("Current state is None. Environment may have ended.")
        
        # Store previous score for reward calculation
        prev_agent_score = self.bank.player_scores[0]
        
        # Get current state
        score = self.current_state["current_score"]
        players_in = self.current_state["players_in"].copy()
        
        # All players decide their actions simultaneously (no look-ahead bias)
        # First, collect decisions from all players using the state BEFORE any decisions are made
        decisions = {}
        
        # Agent decides action (RL agent is always player 0)
        agent_banked_this_turn = False
        if players_in[0]:  # RL agent is still in
            if action == 1:  # bank
                decisions[0] = "bank"
            else:  # roll
                decisions[0] = "roll"
        else:
            decisions[0] = None  # Agent already banked
        
        # Opponents decide their actions using the state BEFORE the agent's decision
        for i in range(1, self.n_players):  # Opponents are players 1+
            if players_in[i]:
                state_for_opponent = {
                    "current_score": score,
                    "rounds_remaining": self.current_state["rounds_remaining"],
                    "player_scores": self.current_state["player_scores"].copy(),
                    "players_in": self.current_state["players_in"].copy(),  # Use original state, not updated
                }
                # Set player_id for opponent if not set
                if not hasattr(self.bank.players[i], 'player_id') or self.bank.players[i].player_id != i:
                    self.bank.players[i].player_id = i
                decisions[i] = self.bank.players[i].decide_action(state_for_opponent)
            else:
                decisions[i] = None  # Opponent already banked
        
        # Now apply all decisions simultaneously
        for i in range(self.n_players):
            if decisions[i] == "bank" and players_in[i]:
                players_in[i] = False
                self.bank.player_scores[i] += score
                if i == 0:  # Track if agent banked
                    agent_banked_this_turn = True
                    self.agent_banked = True
        
        # No intermediate rewards - only reward at game end based on win/loss/tie
        reward = 0.0
        
        # Check if all players have banked
        if all(not player_in for player_in in players_in):
            # Round ends
            terminated = False
            truncated = False
            
            # Update score history
            for i in range(self.n_players):
                self.bank.player_score_history[i].append(self.bank.player_scores[i])
            
            # Move to next round
            self.current_round += 1
            self.round_step = 0
            self.agent_banked = False
            
            if self.current_round > self.rounds:
                # Game over
                terminated = True
                # Only reward here based on win/loss/tie
                reward = self._calculate_final_reward()
                self.current_state = None
                observation = np.zeros(self.observation_space.shape, dtype=np.float32)
                info = {"game_ended": True, "final_scores": self.bank.player_scores.copy()}
                return observation, reward, terminated, truncated, info
            
            # Start new round
            score = self.bank.first_rolls()
            players_in = [True for _ in range(self.n_players)]
            self.current_state = {
                "current_score": score,
                "rounds_remaining": self.rounds - self.current_round + 1,
                "player_scores": self.bank.player_scores.copy(),
                "players_in": players_in.copy(),
            }
            
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, terminated, truncated, info
        
        # Continue the round: roll dice (only if not all banked)
        roll = self.bank.roll()
        
        if sum(roll) == 7:
            # Round ends with score reset to 0 - no intermediate penalty
            reward = 0.0
            
            terminated = False
            truncated = False
            
            # Update score history
            for i in range(self.n_players):
                self.bank.player_score_history[i].append(self.bank.player_scores[i])
            
            # Move to next round
            self.current_round += 1
            self.round_step = 0
            self.agent_banked = False
            
            if self.current_round > self.rounds:
                # Game over
                terminated = True
                # Only reward here based on win/loss/tie
                reward = self._calculate_final_reward()
                self.current_state = None
                observation = np.zeros(self.observation_space.shape, dtype=np.float32)
                info = {"game_ended": True, "final_scores": self.bank.player_scores.copy()}
                return observation, reward, terminated, truncated, info
            
            # Start new round
            score = self.bank.first_rolls()
            players_in = [True for _ in range(self.n_players)]
            self.current_state = {
                "current_score": score,
                "rounds_remaining": self.rounds - self.current_round + 1,
                "player_scores": self.bank.player_scores.copy(),
                "players_in": players_in.copy(),
            }
            
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, terminated, truncated, info
        
        # Update score based on roll
        if roll[0] == roll[1]:
            score *= 2  # Doubles double the score
        else:
            score += sum(roll)
        
        self.round_step += 1
        
        # Check for truncation (round too long)
        truncated = self.round_step >= self.max_round_length
        
        # Update state
        self.current_state = {
            "current_score": score,
            "rounds_remaining": self.current_state["rounds_remaining"],
            "player_scores": self.bank.player_scores.copy(),
            "players_in": players_in.copy(),
        }
        
        observation = self._get_observation()
        info = self._get_info()
        terminated = False
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward for the current step.
        This method is no longer used (rewards calculated in step()),
        kept for compatibility.
        """
        return 0.0
    
    def _calculate_final_reward(self) -> float:
        """
        Calculate final reward at the end of the game.
        Reward: 1.0 if agent wins, 0.5 if tie, 0.0 if loses.
        """
        if self.bank is None:
            return 0.0
        
        agent_score = self.bank.player_scores[0]
        max_opponent_score = max(self.bank.player_scores[1:]) if self.n_opponents > 0 else 0
        
        if agent_score > max_opponent_score:
            return 1.0  # Win
        elif agent_score == max_opponent_score:
            return 0.5  # Tie (points for tying)
        else:
            return 0.0  # Loss
    
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

