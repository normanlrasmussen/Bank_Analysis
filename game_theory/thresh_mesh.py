import numpy as np
import matplotlib.pyplot as plt
from bank import Bank
from players import ThersholdPlayer as ThresholdPlayer
import tqdm
from mpl_toolkits.mplot3d import Axes3D


n = 50

X = np.linspace(50, 1000, n) # Threshold
Z = np.zeros((n, n))

# Create a more informative progress bar
total_iterations = n * n
pbar = tqdm.tqdm(total=total_iterations, 
                 desc="Computing payoff matrix",
                 unit="cell",
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

for i in range(X.shape[0]):
    for j in range(X.shape[0]):
        win_pct, _ = Bank.estimate_win_probability(
            [ThresholdPlayer(threshold=X[i]), ThresholdPlayer(threshold=X[j])], 
            rounds=10, 
            num_simulations=1000
            )
        Z[i, j] = win_pct[0] - win_pct[1]
        # Update progress bar with current values
        pbar.set_postfix({
            'P1': f'{X[i]:.3f}',
            'P2': f'{X[j]:.3f}',
            'Payoff': f'{Z[i, j]:.4f}'
        })
        pbar.update(1)

pbar.close()


# Find all pure Nash equilibria for the zero-sum game
# Player 1 (row i) maximizes, Player 2 (column j) minimizes
print("\n" + "="*60)
print("Finding Pure Nash Equilibria")
print("="*60)

# For each column j, find the row(s) i that maximize Z[i, j] (player 1's best responses)
player1_best_responses = {}
for j in range(n):
    max_val = np.max(Z[:, j])
    best_rows = np.where(Z[:, j] == max_val)[0]
    player1_best_responses[j] = (best_rows, max_val)

# For each row i, find the column(s) j that minimize Z[i, j] (player 2's best responses)
player2_best_responses = {}
for i in range(n):
    min_val = np.min(Z[i, :])
    best_cols = np.where(Z[i, :] == min_val)[0]
    player2_best_responses[i] = (best_cols, min_val)

# Find all pure Nash equilibria: (i, j) where i is best response to j and j is best response to i
nash_equilibria = []
for i in range(n):
    for j in range(n):
        # Check if i is a best response to j (player 1)
        if i in player1_best_responses[j][0]:
            # Check if j is a best response to i (player 2)
            if j in player2_best_responses[i][0]:
                nash_equilibria.append((i, j, Z[i, j], X[i], X[j]))

# Print results
print(f"\nFound {len(nash_equilibria)} pure Nash equilibrium/equilibria:\n")
if len(nash_equilibria) > 0:
    for idx, (i, j, payoff, p1_prob, p2_prob) in enumerate(nash_equilibria, 1):
        print(f"Equilibrium {idx}:")
        print(f"  Strategy profile: (i={i}, j={j})")
        print(f"  Player 1 probability: {p1_prob:.4f}")
        print(f"  Player 2 probability: {p2_prob:.4f}")
        print(f"  Payoff (P1 - P2): {payoff:.6f}")
        print()
else:
    print("No pure Nash equilibria found.")
    print("This game may only have mixed strategy equilibria.")

print("="*60)

np.save('thresh_mesh.npy', Z)
