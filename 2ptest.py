from bank2p import Bank2p
from player import ThersholdPlayer, HumanPlayer

p1 = ThersholdPlayer(threshold=100)
p2 = HumanPlayer()

bank = Bank2p(10, p1, p2, verbose=False)
bank.play_game()