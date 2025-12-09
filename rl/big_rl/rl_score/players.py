import numpy as np

class Player:
    """
    Base class for all players.
    All players must implement the decide_action method.
    """
    def __init__(self, name: str = None):
        self.name = name
    
    def set_player_id(self, player_id: int):
        self.player_id = player_id
    
    def decide_action(self, state):
        raise NotImplementedError("Subclasses must implement this method")

class HumanPlayer(Player):
    """
    Player that is controlled by a human.
    """
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def decide_action(self, state):
        print("Here is the current state:")
        print(f"Rounds remaining: {state['rounds_remaining']}")
        for i in range(len(state["players_in"])):
            if i != self.player_id:
                print(f"Player {i} score: {state['player_scores'][i]}")
        print(f"Your Score: {state['player_scores'][self.player_id]}")
        print(f"Current BANK Value: {state['current_score']}")
        
        while True:
            action = input("Enter your action: ")
            if action == "bank":
                return "bank"
            if action == "roll":
                return "roll"
            print("Invalid action. Please enter 'bank' or 'roll'.")

class ThersholdPlayer(Player):
    """
    Player that banks when the current score is greater than or equal to the threshold.
    """
    def __init__(self, name: str = None, threshold: int = 0):
        super().__init__(name)
        self.threshold = threshold
    
    def decide_action(self, state):
        return "bank" if state["current_score"] >= self.threshold else "roll"

class GreedyPlayer(Player):
    """
    Player doesn't bank until everyone else has banked.
    """
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def decide_action(self, state):
        return "bank" if sum(state["players_in"]) == 1 else "roll"

class GreedyPlayerK(Player):
    """
    Player keeps rolling until everyone else has banked, then takes up to K extra
    rolls before banking. K applies per round, not for the entire game.
    """

    def __init__(self, name: str = None, k: int = 1):
        super().__init__(name)
        if k < 0:
            raise ValueError("k must be non-negative")
        self.k = k
        self._extra_rolls_taken = 0

    def _reset_extra_rolls(self):
        self._extra_rolls_taken = 0

    def decide_action(self, state):
        others_still_in = sum(state["players_in"]) - int(state["players_in"][self.player_id])
        if others_still_in > 0:
            self._reset_extra_rolls()
            return "roll"

        if self._extra_rolls_taken < self.k:
            self._extra_rolls_taken += 1
            return "roll"

        self._reset_extra_rolls()
        return "bank"


class SesquaGreedyPlayer(Player):
    """
    Acts like a greedy player until all opponents bank. Once alone, bank only if
    doing so maintains or takes the lead; otherwise keep rolling.
    """

    def __init__(self, name: str = None):
        super().__init__(name)

    def decide_action(self, state):
        others_still_in = sum(state["players_in"]) - int(state["players_in"][self.player_id])
        if others_still_in > 0:
            return "roll"

        current_score = state["player_scores"][self.player_id]
        scores = state["player_scores"]
        max_other = max(score for idx, score in enumerate(scores) if idx != self.player_id) if len(scores) > 1 else -np.inf
        prospective = current_score + state["current_score"]

        return "bank" if current_score > max_other or prospective > max_other else "roll"
class ProbabilisticPlayer(Player):
    """
    Player that banks with a certain probability.
    """
    def __init__(self, name: str = None, probability: float = 0.5):
        super().__init__(name)
        self.probability = probability
    
    def decide_action(self, state):
        return "bank" if np.random.random() < self.probability else "roll"

class AntiGreedyPlayer(Player):
    """
    Player for 2p games that banks when they would have more points than the opponent.
    """
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def decide_action(self, state):
        if len(state["players_in"]) != 2:
            raise ValueError("AntiGreedyPlayer is only for 2p games")

        prospective_score = state["player_scores"][self.player_id] + state["current_score"]
        return "bank" if prospective_score > state["player_scores"][1 - self.player_id] else "roll"

class TrollPlayer(Player):
    """Player that never banks, rolling until getting a 7."""
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def decide_action(self, state):
        return "roll"