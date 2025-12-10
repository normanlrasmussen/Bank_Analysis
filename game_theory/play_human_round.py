from bank import Bank
from players import RollThresholdPlayer, HumanPlayer, ProbabilisticPlayer, GreedyPlayer, SesquaGreedyPlayer, AntiGreedyPlayer, ThersholdPlayer

players = [
    HumanPlayer(),
    # RollThresholdPlayer(threshold=4),
    # ThersholdPlayer(threshold=100),
    # ProbabilisticPlayer(probability=0.5),
    # GreedyPlayer(),
    # AntiGreedyPlayer(),
    SesquaGreedyPlayer(),
]

rounds = 10

bank = Bank(rounds, players)
bank.play_game()

results = bank.results
print(" ")
if results[0][0] == max(results[0]):
    print("You won!")
else:
    print("You lost!")