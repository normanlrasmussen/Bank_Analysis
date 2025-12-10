from bank import Bank
from players import ThersholdPlayer, ProbabilisticPlayer, RollThresholdPlayer
import matplotlib.pyplot as plt
import numpy as np

rounds = 10
num_simulations = 100000

# thresholds = [50 + 100 * i for i in range(100)]
# thresholds = [5 * i for i in range(100)]
# players = [ThersholdPlayer(threshold=threshold) for threshold in thresholds]

# probs = [0.01 * i for i in range(101)]
# players = [ProbabilisticPlayer(prob) for prob in probs]

# rolls = [i for i in range(1, 101)]
# players = [RollThresholdPlayer(threshold=roll) for roll in rolls]
rolls = [i for i in range(1, 16)]
players = [RollThresholdPlayer(threshold=roll) for roll in rolls]


expected_score, std = Bank.get_expected_value_std(players, rounds, num_simulations=num_simulations)

# for val, threshold in zip(expected_score, thresholds):
#     print(f"Threshold: {threshold}, Expected Score: {val}")



# Plot the expected score vs. std as a scatter plot with color based on threshold
fig, ax = plt.subplots()

# Create scatter plot where color is based on threshold value
scatter = ax.scatter(expected_score, std, c=rolls, cmap='viridis', s=50, edgecolors='none')
ax.set_xlabel('Expected Score')
ax.set_ylabel('Standard Deviation')

plt.title(f"Expected Score vs Standard Deviation (Rounds: {rounds}, Simulations: {num_simulations}, Roll Threshold Players)")
fig.tight_layout()
plt.grid(False)  # Remove grid lines
plt.colorbar(scatter, ax=ax, label='Roll')
plt.savefig(f"figures/expected_score_vs_std_rounds_rolls_{rounds}_simulations_{num_simulations}_paper_1.png")
plt.show()