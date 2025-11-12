from bank import Bank
from player import ThersholdPlayer

thresholds = [50 + 10 * i for i in range(100)]
players = [ThersholdPlayer(threshold=threshold) for threshold in thresholds]

expected_score, winner_count = Bank.estimate_expected_score(players, 10, num_simulations=10000)

for val, threshold, winner_prob in zip(expected_score, thresholds, winner_count):
    print(f"Threshold: {threshold}, Expected Score: {val}, Winner Probability: {winner_prob}")



# Plot the expected score and winner probability vs. threshold
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Expected Score', color=color)
ax1.plot(thresholds, expected_score, marker='o', color=color, label='Expected Score')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Winner Probability', color=color)
ax2.plot(thresholds, winner_count, marker='s', color=color, label='Winner Probability')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Expected Score and Winner Probability vs. Threshold")
fig.tight_layout()
plt.grid(True)
plt.show()
