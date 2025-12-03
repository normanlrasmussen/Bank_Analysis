from bank import Bank
from players import ThersholdPlayer

rounds = 10
num_simulations = 30000

# thresholds = [50 + 500 * i for i in range(100)]
thresholds = [50 + 100 * i for i in range(100)]
# thresholds = [50, 844]
players = [ThersholdPlayer(threshold=threshold) for threshold in thresholds]

expected_score, winner_count = Bank.estimate_expected_score(players, rounds, num_simulations=num_simulations)

for val, threshold in zip(expected_score, thresholds):
    print(f"Threshold: {threshold}, Expected Score: {val}")



# Plot the expected score vs. threshold
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.set_xlabel('Threshold')
ax.set_ylabel('Expected Score')
ax.plot(thresholds, expected_score, marker='o', label='Expected Score')

plt.title(f"Expected Score vs. Threshold (Rounds: {rounds}, Simulations: {num_simulations})")
fig.tight_layout()
plt.grid(True)
plt.savefig(f"figures/expected_score_vs_threshold_rounds_{rounds}_simulations_{num_simulations}.png")
plt.show()