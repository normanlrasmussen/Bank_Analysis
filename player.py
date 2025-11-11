class Player:
    def __init__(self):
        pass
    
    def decide_action(self, bank):
        raise NotImplementedError("Subclasses must implement this method")