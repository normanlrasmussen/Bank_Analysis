import numpy as np
from player import Player

class Bank2p:
    def __init__(self, rounds: int, player_1_strategy: Player, player_2_strategy: Player, verbose: bool = False):
        self.rounds = rounds
        self.player_1 = player_1_strategy
        self.player_2 = player_2_strategy
        self.player_1.set_player_id(0)
        self.player_2.set_player_id(1)
        self.player_1_score = 0
        self.player_2_score = 0
        self.results = None
        self.verbose = verbose

    def roll(self):
        # Roll two dice and return the result
        return np.random.randint(1, 7), np.random.randint(1, 7)
    
    def first_rolls(self):
        # Roll 3 dice and return the sum of the rolls
        rolls = [self.roll() for _ in range(3)] 
        return sum([sum(roll) if sum(roll) != 7 else 70 for roll in rolls])
        
    def play_round(self):
        score = self.first_rolls()
        p1, p2 = True, True
        k = 0
        while True:
            state = {
                "current_score": score,
                "rounds_remaining": self.rounds - self.current_round,
                "player_scores": [self.player_1_score, self.player_2_score],
                "players_in": [p1, p2],
            }
            
            # Get players actions
            p1_action = self.player_1.decide_action(state)
            p2_action = self.player_2.decide_action(state)

            if p1_action == "bank" and p1:
                p1 = False
                self.player_1_score += score
            if p2_action == "bank" and p2:
                p2 = False
                self.player_2_score += score
            
            if not p1 and not p2:
                break
            
            roll = self.roll()
            if sum(roll) == 7:
                break
            elif roll[0] == roll[1]:
                score *= 2
            else:
                score += sum(roll)
            k += 1
        
        return k, score


    def play_game(self):
        self.current_round = 1
        self.player_1_score = 0
        self.player_2_score = 0
        
        while self.current_round <= self.rounds:
            k, score = self.play_round()

            if self.verbose:
                print(f"Round {self.current_round}: Final Score was {score} and number of rolls was {k}")
                print(f"Player 1 score: {self.player_1_score} and Player 2 score: {self.player_2_score}")
                print(" ")

            self.current_round += 1
        
        self.results = [self.player_1_score, self.player_2_score]
