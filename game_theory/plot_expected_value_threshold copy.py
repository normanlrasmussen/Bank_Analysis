from bank import Bank
from players import ThersholdPlayer, ProbabilisticPlayer, RollThresholdPlayer
import matplotlib.pyplot as plt

rounds = 10
num_simulations = 50000

# thresholds = [50 + 100 * i for i in range(100)]
# thresholds = [5 * i for i in range(100)]
# players = [ThersholdPlayer(threshold=threshold) for threshold in thresholds]

# probs = [0.01 * i for i in range(101)]
# players = [ProbabilisticPlayer(prob) for prob in probs]

rolls = [i for i in range(1, 101)]
players = [RollThresholdPlayer(threshold=roll) for roll in rolls]


expected_score, winner_count = Bank.estimate_expected_score(players, rounds, num_simulations=num_simulations)

# for val, threshold in zip(expected_score, thresholds):
#     print(f"Threshold: {threshold}, Expected Score: {val}")



# Plot the expected score vs. threshold
fig, ax = plt.subplots()

ax.set_xlabel('Roll')
ax.set_ylabel('Expected Score')
ax.plot(rolls, expected_score, marker='o', label='Expected Score')

plt.title(f"Expected Score vs. Roll (Rounds: {rounds}, Simulations: {num_simulations})")
fig.tight_layout()
plt.grid(True)
plt.savefig(f"figures/expected_score_vs_roll_rounds_{rounds}_simulations_{num_simulations}_paper.png")
plt.show()