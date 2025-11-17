"""
Animation of threshold player score distributions changing over time.
Shows how distributions evolve as threshold increases from a to b.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys


def roll():
    """Roll two dice and return the result."""
    return np.random.randint(1, 7), np.random.randint(1, 7)


def first_rolls():
    """Roll 3 dice and return the sum of the rolls (7 becomes 70)."""
    rolls = [roll() for _ in range(3)] 
    return sum([sum(roll) if sum(roll) != 7 else 70 for roll in rolls])


def play_round_with_threshold(threshold):
    """
    Play a single round with a threshold player.
    The player banks when the current score >= threshold.
    
    Parameters:
    -----------
    threshold : int
        The threshold at which the player will bank
    
    Returns:
    --------
    int : The score banked in this round (0 if a 7 is rolled before banking)
    """
    # Start with the first 3 rolls
    score = first_rolls()
    
    # Check if we should bank immediately after first 3 rolls
    if score >= threshold:
        return score
    
    # Continue rolling until we hit threshold or a 7
    while True:
        rol = roll()
        dice_sum = sum(rol)
        
        # If a 7 is rolled, score becomes 0 and round ends
        if dice_sum == 7:
            return 0
        
        # If doubles (same value on both dice), double the score
        if rol[0] == rol[1]:
            score *= 2
        else:
            # Otherwise, add the sum to the score
            score += dice_sum
        
        # Check if we should bank now
        if score >= threshold:
            return score


def play_game_with_threshold(threshold, n_rounds):
    """
    Play a full game (multiple rounds) with a threshold player.
    
    Parameters:
    -----------
    threshold : int
        The threshold at which the player will bank in each round
    n_rounds : int
        Number of rounds to play
    
    Returns:
    --------
    int : The total score after all rounds
    """
    total_score = 0
    for _ in range(n_rounds):
        round_score = play_round_with_threshold(threshold)
        total_score += round_score
    return total_score


def animate_threshold_distributions(threshold_start, threshold_end, n_rounds=10, 
                                    n_trials=500, n_frames=50, bins=50, 
                                    interval=200, figsize=(12, 8)):
    """
    Create an animation showing how score distributions change as threshold increases.
    
    Parameters:
    -----------
    threshold_start : int
        Starting threshold value
    threshold_end : int
        Ending threshold value
    n_rounds : int, default=10
        Number of rounds per game
    n_trials : int, default=500
        Number of game simulations per threshold
    n_frames : int, default=50
        Number of animation frames (thresholds to show)
    bins : int, default=50
        Number of bins for the histogram
    interval : int, default=200
        Animation interval in milliseconds
    figsize : tuple, default=(12, 8)
        Figure size (width, height) in inches
    
    Returns:
    --------
    FuncAnimation : The animation object
    """
    # Generate threshold values to animate over
    thresholds = np.linspace(threshold_start, threshold_end, n_frames, dtype=int)
    
    # Pre-compute all distributions
    print("Pre-computing distributions...")
    all_results = {}
    all_scores = []
    
    for threshold in thresholds:
        print(f"Computing threshold={threshold}...", end='\r')
        final_scores = np.array([play_game_with_threshold(threshold, n_rounds) 
                                 for _ in range(n_trials)])
        all_results[threshold] = final_scores
        all_scores.extend(final_scores)
    
    print(f"\nDone! Computed {len(thresholds)} distributions.")
    
    # Determine common bin edges for all histograms
    min_score = min(all_scores)
    max_score = max(all_scores)
    bin_edges = np.linspace(min_score, max_score, bins + 1)
    
    # Find the maximum frequency across all distributions for constant y-axis
    print("Computing maximum frequency for constant y-axis...")
    max_frequency = 0
    for threshold in thresholds:
        scores = all_results[threshold]
        counts, _ = np.histogram(scores, bins=bin_edges)
        max_frequency = max(max_frequency, max(counts) if len(counts) > 0 else 0)
    
    # Set y-axis limit with some padding
    y_max = max_frequency * 1.1 if max_frequency > 0 else 10
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Initialize histogram (will be updated in animation)
    n, bins_plot, patches = ax.hist([], bins=bin_edges, alpha=0.6, 
                                     color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Set up labels and title
    ax.set_xlabel('Final Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_xlim(min_score, max_score)
    ax.set_ylim(0, y_max)  # Constant y-axis based on max frequency
    ax.grid(True, alpha=0.3)
    
    # Text for threshold and statistics
    threshold_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                             verticalalignment='top', horizontalalignment='left',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                             fontsize=12, fontweight='bold')
    
    stats_text = ax.text(0.98, 0.98, '', transform=ax.transAxes,
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                         fontsize=10, family='monospace')
    
    def animate(frame):
        """Update the histogram for each frame."""
        threshold = thresholds[frame]
        scores = all_results[threshold]
        
        # Update histogram data
        counts, _ = np.histogram(scores, bins=bin_edges)
        
        # Update the histogram bars
        for count, patch in zip(counts, patches):
            patch.set_height(count)
        
        # Update threshold text
        threshold_text.set_text(f'Threshold: {threshold}')
        
        # Update statistics
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        std_score = np.std(scores)
        zero_percent = 100 * np.sum(scores == 0) / len(scores)
        
        stats_str = (
            f'Mean: {mean_score:.1f}\n'
            f'Median: {median_score:.1f}\n'
            f'Std: {std_score:.1f}\n'
            f'Zero: {zero_percent:.1f}%'
        )
        stats_text.set_text(stats_str)
        
        # Update title
        ax.set_title(f'Threshold Player Score Distribution Animation\n'
                    f'Rounds: {n_rounds}, Trials: {n_trials}, Frame: {frame+1}/{n_frames}',
                    fontsize=14, fontweight='bold')
        
        return patches, threshold_text, stats_text
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=interval, 
                        blit=False, repeat=True)
    
    plt.tight_layout()
    return anim


def main():
    """Main function to run the animation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Animate threshold player distributions')
    parser.add_argument('--start', type=int, default=50, 
                       help='Starting threshold (default: 50)')
    parser.add_argument('--end', type=int, default=500, 
                       help='Ending threshold (default: 200)')
    parser.add_argument('--rounds', type=int, default=10, 
                       help='Number of rounds per game (default: 10)')
    parser.add_argument('--trials', type=int, default=500, 
                       help='Number of trials per threshold (default: 500)')
    parser.add_argument('--frames', type=int, default=50, 
                       help='Number of animation frames (default: 50)')
    parser.add_argument('--interval', type=int, default=200, 
                       help='Animation interval in milliseconds (default: 200)')
    parser.add_argument('--save', type=str, default='animation.gif', 
                       help='Save animation to file (default: animation.gif)')
    
    args = parser.parse_args()

    anim = animate_threshold_distributions(
        threshold_start=args.start,
        threshold_end=args.end,
        n_rounds=args.rounds,
        n_trials=args.trials,
        n_frames=args.frames,
        interval=args.interval
    )
    
    if args.save:
        print(f"Saving animation to {args.save}...")
        anim.save(args.save, writer='pillow', fps=1000//args.interval)
        print("Animation saved!")
    else:
        print("Displaying animation. Close the window to exit.")
        plt.show()


if __name__ == '__main__':
    main()

