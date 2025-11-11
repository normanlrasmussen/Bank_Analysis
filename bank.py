import numpy as np

from player import Player

class Bank2p:
    def __init__(self, rounds: int, player_1_strategy: Player, player_2_strategy: Player):
        self.rounds = rounds
        self.player_1 = player_1_strategy
        self.player_2 = player_2_strategy
        self.player_1_score = 0
        self.player_2_score = 0

    def roll(self):
        return np.random.randint(1, 7), np.random.randint(1, 7)
    
    def first_rolls(self):
        rolls = [self.roll() for _ in range(3)] 
        return sum([sum(roll) if sum(roll) != 7 else 70 for roll in rolls])

    def regular_roll(self):
        pass

    def play_round(self):

        # Make 3 rolls at the beginning of the round
        score = self.first_rolls()

        while True:
            
        
bank = Bank2p(10, Player(), Player())
print(bank.first_rolls())