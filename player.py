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