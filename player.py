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
        for i, player in enumerate(state["players_in"]):
            if i == self.player_id:
                continue
            else:
                print(f"Player {i} score: {state['player_scores'][i]}")
        print(f"Your Score: {state['player_scores'][self.player_id]}")
        print(f"Current BANK Value: {state['current_score']}")
        
        while True:
            action = input("Enter your action: ")
            if action == "bank":
                return "bank"
            elif action == "roll":
                return "roll"
            else:
                print("Invalid action. Please enter 'bank' or 'roll'.")

class ThersholdPlayer(Player):
    """
    Player that banks when the current score is greater than or equal to the threshold.
    """
    def __init__(self, name: str = None, threshold: int = 0):
        super().__init__(name)
        self.threshold = threshold
    
    def decide_action(self, state):
        if state["current_score"] >= self.threshold:
            return "bank"
        else:
            return "roll"

class GreedyPlayer(Player):
    """
    Player doesn't bank until everyone else has banked.
    """
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def decide_action(self, state):
        if sum(state["players_in"]) == 1:
            return "bank"
        else:
            return "roll"

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
            # Still waiting for the rest to bank.
            self._reset_extra_rolls()
            return "roll"

        # All other players have banked.
        if self._extra_rolls_taken < self.k:
            self._extra_rolls_taken += 1
            return "roll"

        self._reset_extra_rolls()
        return "bank"