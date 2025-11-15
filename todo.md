=========================================================
BANK SELF-PLAY REINFORCEMENT LEARNING PROJECT PLAN
=========================================================

Goal:
-----
Teach a bot to play the dice game "Bank" using self-play reinforcement learning (RL).
We’ll start simple (tabular or DQN) and move toward neural network training.

---------------------------------------------------------
PHASE 1: SETUP
---------------------------------------------------------
1. Organize your files:
   /bank_rl/
     ├── bank.py             # Your Bank environment (already done)
     ├── player.py           # Base Player class
     ├── rl_player.py        # Reinforcement learning agent
     ├── train_selfplay.py   # Training loop
     ├── utils.py            # optional: helper functions
     ├── results/            # for saving models and plots

2. Install dependencies:
   pip install numpy torch matplotlib tqdm

3. Make sure bank.py runs with random/static players first.

---------------------------------------------------------
PHASE 2: ENVIRONMENT CLEANUP
---------------------------------------------------------
✅ Add a method in Bank to return:
   - current state vector
   - possible actions (bank or roll)
   - reward signal

Example state vector (normalized):
   [current_score/100, rounds_remaining/total_rounds, 
    player_score/500, is_player_in]

✅ Modify Bank to allow step-by-step play rather than whole game play_game().
   (You want env.reset(), env.step(action), env.done)

---------------------------------------------------------
PHASE 3: BASIC SELF-PLAY STRUCTURE
---------------------------------------------------------
1. Create an RLPlayer class (inherits Player):
   - Has epsilon-greedy policy for exploration
   - Has replay memory to store (state, action, reward, next_state)
   - Uses small MLP (2 hidden layers of 64 units) to predict Q-values

2. Self-play setup:
   - Two players share the same RLPlayer model
   - Each game: alternate who goes first
   - Collect experience from both players into the same replay buffer

3. Training loop (pseudo-code):
   for episode in range(num_episodes):
       env = Bank(rounds=5, players=[agent, agent.clone()])
       state = env.reset()
       done = False
       while not done:
           action = agent.act(state)
           next_state, reward, done, info = env.step(action)
           agent.remember(state, action, reward, next_state, done)
           agent.learn_from_memory(batch_size=64)
           state = next_state
       log episode reward

---------------------------------------------------------
PHASE 4: EVALUATION
---------------------------------------------------------
1. After every 100 episodes:
   - Play 50 games vs. a "greedy" baseline (banks at 30+ points)
   - Compute average score and win rate

2. Plot learning curve (episode vs. average reward)
   Save with matplotlib

---------------------------------------------------------
PHASE 5: EXPERIMENTS TO TRY
---------------------------------------------------------
🧪 Try these in order:

[1] Change reward signals:
    - +1 if win, -1 if lose
    - Incremental reward = score gained that turn

[2] Exploration tuning:
    - epsilon decay from 1.0 → 0.1 over 10k episodes

[3] Compare architectures:
    - Small MLP vs. larger (128x128)
    - Add dropout or layer normalization

[4] Add population self-play:
    - Train against a mix of older saved versions of the agent

[5] Record gameplay:
    - Track decisions (bank/roll) vs. score curve to visualize strategy evolution

---------------------------------------------------------
PHASE 6: OPTIONAL ADVANCED IDEAS
---------------------------------------------------------
🌟 Use Policy Gradient (PPO or REINFORCE)
   - Replace Q-learning with a policy network π(a|s)
   - Reward = final score difference
   - Easier to handle stochastic environments

🌟 Use Fictitious Self-Play (for equilibrium strategies)
   - Maintain pool of policies and train against random opponent from the pool

🌟 Add network visualization and action heatmaps

---------------------------------------------------------
PHASE 7: SAVING & DEPLOYMENT
---------------------------------------------------------
- Save models: torch.save(agent.q_net.state_dict(), "results/best_model.pt")
- Save plots: plt.savefig("results/learning_curve.png")
- Optional: implement "human vs bot" mode to test yourself.

=========================================================
SUMMARY CHECKLIST
=========================================================
☐ Bank environment allows step-based play
☐ RLPlayer implemented with Q-network
☐ Self-play loop runs end-to-end
☐ Rewards defined and stable
☐ Learning curve improves over time
☐ Model saved and visualized
☐ Baseline comparison (greedy player)
☐ Optional: PPO / policy gradient version
=========================================================

