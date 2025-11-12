from bank2p import Bank2p
from player import ThersholdPlayer

p1 = ThersholdPlayer(threshold=100)
p2 = ThersholdPlayer(threshold=200)

bank = Bank2p(10, p1, p2, verbose=True)
bank.play_game()