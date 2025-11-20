from bank import Bank
from player import ThersholdPlayer

players = [ThersholdPlayer(threshold=(50 + 100 * i)) for i in range(10)]

bank = Bank(1000, players)
bank.play_game()
bank.plot_player_scores()