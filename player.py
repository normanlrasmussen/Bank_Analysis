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