import copy
import stat
import numpy as np
import matplotlib.pyplot as plt

from players import Player

class Bank:
    def __init__(self, rounds: int, players: list[Player], verbose: bool = False):
        self.rounds = rounds
        self.players = players
        for i, player in enumerate(self.players):
            player.set_player_id(i)
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
        players_in = [True for _ in range(len(self.players))]
        k = 0
        while True:
            state = {
                "current_score": score,
                "rounds_remaining": self.rounds - self.current_round,
                "player_scores": self.player_scores.copy(),
                "players_in": players_in.copy(),
            }
            
            # Collect all player decisions simultaneously (no look-ahead bias)
            # All players see the same state before any decisions are made
            decisions = {}
            for i, player in enumerate(self.players):
                if players_in[i]:
                    decisions[i] = player.decide_action(state)
                else:
                    decisions[i] = None  # Player already banked
            
            # Apply all decisions simultaneously
            for i, action in decisions.items():
                if action == "bank" and players_in[i]:
                    players_in[i] = False
                    self.player_scores[i] += score

            if all(not player_in for player_in in players_in):
                break
            
            roll = self.roll()
            if sum(roll) == 7:
                break
            elif roll[0] == roll[1]:
                score *= 2
            else:
                score += sum(roll)
            k += 1
        
        # update player score history
        for i in range(len(self.players)):
            self.player_score_history[i].append(self.player_scores[i])

        return k, score

    def play_game(self):
        self.current_round = 1
        self.player_scores = [0 for _ in range(len(self.players))]
        self.player_score_history = [[] for _ in range(len(self.players))]
        
        while self.current_round <= self.rounds:
            k, score = self.play_round()

            if self.verbose:
                print(f"Round {self.current_round}: Final Score was {score} and number of rolls was {k}")
                for i, player in enumerate(self.players):
                    print(f"Player {i} score: {self.player_scores[i]}")
                print(" ")

            self.current_round += 1
        
        self.results = self.player_scores, self.player_score_history

    def plot_player_scores(self, filename: str = None):
        for i, player in enumerate(self.players):
            plt.plot(self.player_score_history[i], label=f"Player {i}", marker='o')
        plt.legend()
        plt.grid(True)
        plt.xlabel("Round")
        plt.ylabel("Score")
        plt.title(f"Player Scores for {self.rounds} rounds")

        # Use argmax on self.player_scores to determine the winner
        winner = np.argmax(self.player_scores)
        winner_text = f"Winner: Player {winner}"
        plt.gcf().text(0.01, 0.97, winner_text, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    @staticmethod
    def get_expected_value(players: list[Player], rounds: int, num_simulations: int = 1000):
        expected_value = np.zeros(len(players))
        for _ in range(num_simulations):
            fresh_players = [copy.deepcopy(player) for player in players]
            bank = Bank(rounds, fresh_players)
            bank.play_game()
            expected_value += np.array(bank.player_scores)
        return expected_value / num_simulations

    @staticmethod # TODO: Depreciate this method
    def estimate_expected_score(players: list[Player], rounds: int, num_simulations: int = 1000):
        expected_score = np.zeros(len(players))
        winner_count = np.zeros(len(players))
        for _ in range(num_simulations):
            fresh_players = [copy.deepcopy(player) for player in players]
            bank = Bank(rounds, fresh_players)
            bank.play_game()
            expected_score += np.array(bank.player_scores)
            winner_count += np.array(bank.player_scores) == max(bank.player_scores)
        return expected_score / num_simulations, winner_count / num_simulations

    @staticmethod
    def estimate_win_probability(players: list[Player], rounds: int, num_simulations: int = 1000):
        win_probability = np.zeros(len(players))
        tie_probability = np.zeros(len(players))
        for _ in range(num_simulations):
            fresh_players = [copy.deepcopy(player) for player in players]
            bank = Bank(rounds, fresh_players)
            bank.play_game()
            scores = bank.player_scores
            max_score = max(scores)
            if scores.count(max_score) == 1:
                win_probability[scores.index(max_score)] += 1
            elif scores.count(max_score) == len(players):
                tie_probability += 1
        return win_probability / num_simulations, tie_probability / num_simulations

    @staticmethod
    def get_all(players: list, rounds: int, num_simulations: int = 1000):
        """
        Simulate games and return expected values, win percentage, and tie probability.
        Returns:
            expected_score: np.array of expected scores per player
            win_percentage: np.array of probability of winning per player
            ties_percentage: np.array of probability of each player participating in a tie
        """
        expected_score = np.zeros(len(players))
        win_count = np.zeros(len(players))
        tie_count = np.zeros(len(players))

        for _ in range(num_simulations):
            fresh_players = [copy.deepcopy(player) for player in players]
            bank = Bank(rounds, fresh_players)
            bank.play_game()
            scores = bank.player_scores
            max_score = max(scores)
            winners = [i for i, s in enumerate(scores) if s == max_score]
            expected_score += np.array(scores)
            if len(winners) == 1:
                win_count[winners[0]] += 1
            if len(winners) > 1:
                for winner in winners:
                    tie_count[winner] += 1

        expected_score /= num_simulations
        win_percentage = win_count / num_simulations
        ties_percentage = tie_count / num_simulations

        return expected_score, win_percentage, ties_percentage