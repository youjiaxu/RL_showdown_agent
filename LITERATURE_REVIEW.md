# Reinforcement Learning for Pokémon Showdown: Literature Review

## Available RL Algorithms in CARES Library

Your project has access to the CARES Reinforcement Learning library with the following algorithms:

### Value-Based Methods (Discrete Action Spaces)
1. **DQN** - Deep Q-Network (Mnih et al., 2015)
2. **DoubleDQN** - Double Deep Q-Network (van Hasselt et al., 2016)
3. **DuelingDQN** - Dueling Network Architecture (Wang et al., 2016)
4. **PERDQN** - Prioritized Experience Replay DQN (Schaul et al., 2016)
5. **NoisyNet** - Noisy Networks for Exploration (Fortunato et al., 2018)
6. **C51** - Categorical DQN (Bellemare et al., 2017)
7. **QRDQN** - Quantile Regression DQN (Dabney et al., 2018)
8. **Rainbow** - Combines multiple DQN improvements (Hessel et al., 2018)

### Policy-Based Methods (Continuous Actions - Not Applicable)
- TD3, SAC, PPO, DDPG variants (designed for continuous control)

**Recommended for Pokémon Showdown:** DQN, DoubleDQN, DuelingDQN, Rainbow

---

## 1. Introduction to Reinforcement Learning

### 1.1 Fundamentals of RL

Reinforcement Learning is a paradigm of machine learning where an agent learns to make sequential decisions by interacting with an environment (Sutton & Barto, 2018). Unlike supervised learning, RL does not require labeled examples but instead learns from the consequences of actions through a reward signal.

The core components of an RL system are:

1. **Agent**: The learner/decision-maker
2. **Environment**: The world the agent interacts with
3. **State (s)**: The current situation of the environment
4. **Action (a)**: Choices available to the agent
5. **Reward (r)**: Scalar feedback signal
6. **Policy (π)**: Mapping from states to actions

The agent's objective is to learn a policy π that maximizes the expected cumulative reward, formally defined as:

```
J(π) = E[∑(γ^t * r_t)]
```

where γ ∈ [0,1] is the discount factor determining the importance of future rewards.

### 1.2 The Markov Decision Process (MDP)

RL problems are formalized as Markov Decision Processes, defined by the tuple (S, A, P, R, γ):

- **S**: State space
- **A**: Action space  
- **P**: State transition probability P(s'|s,a)
- **R**: Reward function R(s,a,s')
- **γ**: Discount factor

The **Markov property** states that the future depends only on the current state, not the history:

```
P(s_t+1 | s_t, a_t, s_t-1, a_t-1, ..., s_0, a_0) = P(s_t+1 | s_t, a_t)
```

This property is crucial for tractable learning but can be limiting in partially observable environments.

### 1.3 Value Functions and Q-Learning

The **state-value function** V^π(s) represents the expected return starting from state s under policy π:

```
V^π(s) = E_π[∑(γ^t * r_t) | s_0 = s]
```

The **action-value function** Q^π(s,a) represents the expected return of taking action a in state s, then following π:

```
Q^π(s,a) = E_π[∑(γ^t * r_t) | s_0 = s, a_0 = a]
```

**Q-Learning** (Watkins & Dayan, 1992) is a model-free RL algorithm that learns Q*(s,a) - the optimal action-value function - using the Bellman optimality equation:

```
Q*(s,a) = E[r + γ * max_a' Q*(s',a')]
```

The update rule is:

```
Q(s,a) ← Q(s,a) + α[r + γ * max_a' Q(s',a') - Q(s,a)]
```

where α is the learning rate.

---

## 2. Deep Q-Networks (DQN)

### 2.1 The DQN Revolution

Traditional Q-learning uses tabular representations, infeasible for high-dimensional state spaces. **Deep Q-Networks** (Mnih et al., 2015) revolutionized RL by approximating Q(s,a) with a deep neural network, enabling learning from raw pixels in Atari games.

**Key Innovation #1: Experience Replay**

DQN stores experiences e_t = (s_t, a_t, r_t, s_t+1) in a replay buffer D. During training, mini-batches are sampled uniformly to break temporal correlations:

```python
# Pseudocode
for episode in episodes:
    for step in episode:
        observe (s_t, a_t, r_t, s_t+1)
        store transition in D
        
        sample mini-batch from D
        compute target: y_i = r_i + γ * max_a' Q(s_i', a'; θ^-)
        perform gradient descent on (y_i - Q(s_i, a_i; θ))^2
```

**Benefits:**
- Breaks correlation between consecutive samples
- Enables learning from rare events multiple times
- Improves data efficiency

**Key Innovation #2: Target Network**

DQN uses a separate target network θ^- that is updated periodically (every C steps), while the main network θ is updated every step. This stabilizes learning by preventing the "moving target" problem.

### 2.2 DQN Variants and Improvements

**Double DQN (van Hasselt et al., 2016)**

Addresses overestimation bias in standard DQN by decoupling action selection from evaluation:

```
y_t = r_t + γ * Q(s_t+1, argmax_a Q(s_t+1, a; θ); θ^-)
```

Action selection uses online network θ, evaluation uses target network θ^-.

**Dueling DQN (Wang et al., 2016)**

Separates state-value V(s) and advantage function A(s,a):

```
Q(s,a) = V(s) + (A(s,a) - mean_a' A(s,a'))
```

This architecture learns which states are valuable independent of actions, beneficial when many actions have similar values.

**Prioritized Experience Replay (Schaul et al., 2016)**

Samples transitions proportional to their TD-error magnitude:

```
P(i) = p_i^α / ∑_k p_k^α
```

where p_i = |δ_i| + ε (TD-error + small constant). High TD-error transitions are sampled more frequently, focusing learning on "surprising" events.

**Noisy Networks (Fortunato et al., 2018)**

Adds parametric noise to network weights for exploration:

```
y = (μ^w + σ^w ⊙ ε^w)x + μ^b + σ^b ⊙ ε^b
```

Replaces ε-greedy exploration with learned exploration strategy.

**Rainbow DQN (Hessel et al., 2018)**

Combines six extensions:
1. Double Q-Learning
2. Prioritized Replay
3. Dueling Architecture
4. Multi-step Learning
5. Distributional RL (C51)
6. Noisy Networks

Rainbow achieves state-of-the-art performance on Atari, demonstrating that these improvements are complementary.

---

## 3. Reinforcement Learning in Games

### 3.1 Historical Context

Games have been a proving ground for AI since the inception of the field:

- **Chess**: Deep Blue defeated Kasparov (1997) using minimax search
- **Go**: AlphaGo defeated Lee Sedol (2016) using deep RL + Monte Carlo Tree Search (Silver et al., 2016)
- **Atari**: DQN achieved human-level performance (2015) on 49 games (Mnih et al., 2015)
- **StarCraft II**: AlphaStar defeated professional players (2019) using multi-agent RL (Vinyals et al., 2019)

### 3.2 Game Characteristics and RL Challenges

| Game Type | State Space | Action Space | Observable | Example |
|-----------|-------------|--------------|------------|---------|
| **Deterministic, Perfect Info** | Small-Large | Small-Large | Fully | Chess, Go |
| **Stochastic, Perfect Info** | Large | Medium | Fully | Backgammon |
| **Stochastic, Imperfect Info** | Huge | Medium-Large | Partially | Poker, Pokémon |
| **Real-time, Partial Info** | Continuous | Continuous | Partially | StarCraft II |

**Pokémon Showdown** falls into the "Stochastic, Imperfect Information" category, presenting unique challenges:

1. **Partial Observability**: Opponent's team/moves unknown until revealed
2. **Stochasticity**: Move accuracy, critical hits, damage rolls introduce randomness
3. **High Branching Factor**: 4 moves × 5 switches = 9 actions per turn
4. **Long Horizons**: Battles last 10-50 turns
5. **Strategic Depth**: Type matchups, stat stages, entry hazards, status conditions

---

## 4. Pokémon Showdown and Reinforcement Learning

### 4.1 Pokémon as an RL Domain

Pokémon battles are turn-based strategy games with the following properties:

**Game Mechanics:**
- **Type Effectiveness**: 18 types with complex interaction matrix (0x, 0.5x, 1x, 2x, 4x damage)
- **Stats**: HP, Attack, Defense, Sp.Atk, Sp.Def, Speed (+stat stages -6 to +6)
- **Status Conditions**: Burn, Paralyze, Sleep, Freeze, Poison
- **Entry Hazards**: Stealth Rock, Spikes, Toxic Spikes, Sticky Web
- **Weather**: Sun, Rain, Sand, Hail (affect move power/accuracy)
- **Abilities**: Special effects (e.g., Levitate grants Ground immunity)
- **Items**: Held items modify stats/abilities (e.g., Life Orb, Leftovers)

**Strategic Complexity:**
- **Switch Decisions**: When to switch vs. attack
- **Move Selection**: Which of 4 moves to use
- **Setup Opportunities**: Using stat-boost moves when safe
- **Prediction**: Anticipating opponent switches/moves
- **Resource Management**: PP (move uses), HP preservation

### 4.2 Prior Work on Pokémon AI

**Rule-Based Approaches:**
- **MaxBasePowerPlayer**: Always uses highest base power move
- **SimpleHeuristicsPlayer**: Uses type effectiveness heuristics
- **Limitations**: Cannot learn, poor adaptation, exploitable

**Search-Based Approaches:**
- **Minimax/Expectimax**: Evaluate game trees with damage calculations
- **Monte Carlo Tree Search**: Simulate random playouts
- **Challenges**: Imperfect information, high branching factor, stochasticity

**Machine Learning Approaches:**
- **Supervised Learning**: Learn from human replays (limited by data quality)
- **Reinforcement Learning**: Learn through self-play (our approach)

### 4.3 Existing RL Implementations

**PS-AI (Pokémon Showdown AI):**
- Uses DQN with custom state representation
- Encodes moves, types, stats, hazards as feature vector
- Reward based on HP differential
- Challenges: Sparse rewards, credit assignment

**PokéLLMon (Hu et al., 2024):**
- Uses LLMs (GPT-4) with in-context learning
- Provides battle state as text, receives action as text
- Achieves 49% win rate vs. ladder players
- Limitation: Slow inference, expensive API calls

**Key Insights from Literature:**
1. **State Representation Matters**: Encoding domain knowledge (type charts, abilities) improves learning
2. **Reward Shaping Critical**: Sparse win/loss rewards too slow; need intermediate signals
3. **Exploration Important**: ε-greedy insufficient; need strategic exploration
4. **Temporal Credit Assignment Hard**: Long battles make it difficult to attribute success to specific actions

---

## 5. State Representation Design

### 5.1 Principles of State Design

An effective state representation must be:

1. **Markov**: Captures sufficient information for decision-making
2. **Compact**: Avoids curse of dimensionality
3. **Informative**: Contains features relevant to optimal policy
4. **Normalized**: Prevents gradient issues in neural networks

### 5.2 Feature Categories for Pokémon

**Core Battle State (71 features):**
```
Health Information (12):
- Team HP fractions: 6 values (one per Pokémon)
- Opponent HP fractions: 6 values

Active Pokémon Stats (22):
- Normalized stats: 5 + 5 (attack, defense, sp.atk, sp.def, speed)
- Status conditions: 1 + 1 (encoded as 0-5 for None/BRN/PAR/SLP/FRZ/PSN)
- Stat stages: 5 + 5 (boost levels -6 to +6, normalized)
- Speed advantage: 1 (binary: faster or not)

Move Information (28):
- For each of 4 moves × 7 features:
  * Base power (normalized)
  * Type effectiveness (0-4x range)
  * Estimated damage
  * Priority (-5 to +5)
  * Category (Physical/Special/Status flags)

Battle Context (8):
- Fainted counts: 2 (normalized to 0-6)
- Turn number: 1 (normalized to 0-50)
- Weather: 4 (one-hot: sun, rain, sand, hail)
- Available switches: 1 (0-5 normalized)
```

**Temporal Context (22 features - V2 Enhancement):**
```
Action History (9):
- Last 3 actions × 3 features:
  * Action ID (normalized)
  * Action type (move/switch encoded)
  * Recency (turns ago)

Strategic Indicators (13):
- Battle phase: 1 (early/mid/late game 0-1)
- Momentum: 2 (damage dealt/taken trends)
- Boost efficiency: 5 (room for stat improvements)
- Action patterns: 2 (recent move/switch counts)
- Game phase: 3 (early/mid/late flags)
```

**Field Awareness (8 features - V3 Enhancement):**
```
Entry Hazards (8):
- Opponent hazards: 4 (Stealth Rock, Spikes, T.Spikes, Sticky Web)
- Our hazards: 4 (same)
```

**Opponent Intelligence (6 features - V3 Enhancement):**
```
Ability Indicators (3):
- Has immunity ability: 1 (Volt Absorb, Levitate, etc.)
- Has stat-boost ability: 1 (Intimidate, Defiant, etc.)
- Has weather ability: 1 (Drizzle, Drought, etc.)

Item Indicators (3):
- Has HP recovery item: 1 (Leftovers, Sitrus Berry, etc.)
- Has power item: 1 (Life Orb, Choice Band, etc.)
- Has defensive item: 1 (Assault Vest, Rocky Helmet, etc.)
```

**Total State Size: 112 features**

### 5.3 Design Rationale

**Why These Features?**

1. **Type Effectiveness**: Encoded per move rather than globally; agent learns context-dependent usage
2. **Stat Stages**: Crucial for understanding setup strategies (e.g., +2 Attack Dragon Dance)
3. **Momentum**: Helps agent recognize winning/losing streaks
4. **Hazards**: Critical for learning not to spam Stealth Rock when already up
5. **Ability/Item Indicators**: Prevents agent from using Electric moves vs. Volt Absorb

**What's Missing?**
- Team preview information (opponent's full team)
- Move PP (remaining uses)
- Specific move IDs (learns implicitly through effectiveness)
- Detailed ability/item names (learns categories)

---

## 6. Reward Function Design

### 6.1 The Reward Shaping Challenge

Reward design is the most critical and challenging aspect of RL. A poorly designed reward function leads to:

- **Sparse Rewards**: Agent receives signal only at battle end; slow learning
- **Dense but Misleading**: Agent exploits loopholes ("reward hacking")
- **Fragile**: Small changes cause catastrophic policy collapse

**Example of Reward Hacking:**
```python
# BAD: Reward stat boosts directly
reward += 0.5 if action_is_stat_boost else 0.0

# Result: Agent spams Swords Dance 50 times, never attacks
```

### 6.2 Reward Function Evolution

**V1: Damage-Only Reward (Naive)**
```python
reward = (opponent_hp_lost - our_hp_lost) * 0.5
```

**Problems:**
- Encourages reckless damage trades
- No incentive to win efficiently
- Ignores strategic positioning (hazards, stat boosts)

**V2: Multi-Component Reward (Fragile)**
```python
reward = (
    damage_reward +        # HP differential
    ko_reward +            # Knockout bonuses
    strategic_reward +     # Setup moves
    coherence_reward +     # Strategic sequences
    adaptation_reward +    # Learning patterns
    efficiency_reward +    # Win quickly
    exploration_reward     # Action variety
)
```

**Problems:**
- 7 components, each with multiple sub-rewards
- Magnitudes mismatched (KO reward dominated)
- Agent exploited specific components
- Extremely fragile to hyperparameter changes

**V3: Conservative Tri-Component (Current)**
```python
# Component 1: Dense damage signal
damage_reward = (current_advantage - prior_advantage) * 0.5
damage_reward = clip(damage_reward, -0.5, +0.5)

# Component 2: Sparse KO events
opponent_kos = current_fainted_opp - prior_fainted_opp
if opponent_kos > 0:
    stage_multiplier = 1.0 + (6 - remaining_opponents) * 0.05
    ko_reward += 1.0 * stage_multiplier
ko_reward = clip(ko_reward, -2.0, +2.0)

# Component 3: Terminal outcome
if battle_finished:
    outcome_reward = +2.0 if won else -2.0

total_reward = damage_reward + ko_reward + outcome_reward
total_reward = clip(total_reward, -3.0, +3.0)
```

**Design Principles:**
1. **Conservative Magnitudes**: Damage ±0.5, KO ±2.0, Outcome ±2.0
2. **Tight Clipping**: Prevents any component from dominating
3. **Late-Game Scaling**: KOs more valuable when fewer opponents remain
4. **No Explicit Strategy Rewards**: Let agent discover through outcome

### 6.3 State-Reward Alignment

**Critical Principle**: Agent must observe in state what it's rewarded for.

**Example - Hazard Spam Prevention:**

**WRONG (Misaligned):**
```python
# State: No hazard visibility
state = [health, stats, moves, ...]  # Missing hazards!

# Reward: Penalty for redundant hazards
if action == "Stealth Rock" and hazards_already_up:
    reward -= 0.3  # Agent can't learn from this!
```

**RIGHT (Aligned):**
```python
# State: Hazard visibility
state = [..., opponent_hazards, our_hazards]  # Can see hazards!

# Reward: Penalty for redundant hazards
if action == "Stealth Rock" and state[hazard_idx] == 1.0:
    reward -= 0.3  # Agent can now correlate state → action → penalty
```

---

## 7. Training Methodology

### 7.1 DQN Training Pipeline

```python
# Pseudocode for Pokémon Showdown DQN training

class PokemonDQN:
    def __init__(self):
        self.q_network = NeuralNetwork(state_dim=112, action_dim=9)
        self.target_network = copy(self.q_network)
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.epsilon = 1.0  # Exploration rate
        
    def train(self, num_episodes=10000):
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            
            while not done:
                # ε-greedy action selection
                if random() < self.epsilon:
                    action = random_action()
                else:
                    action = argmax(self.q_network(state))
                
                # Execute action
                next_state, reward, done = env.step(action)
                
                # Store transition
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # Sample mini-batch and train
                if len(self.replay_buffer) > batch_size:
                    batch = self.replay_buffer.sample(batch_size)
                    loss = self.compute_td_loss(batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                state = next_state
                
            # Decay exploration
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            # Update target network periodically
            if episode % 100 == 0:
                self.target_network = copy(self.q_network)
```

### 7.2 Hyperparameters

Critical hyperparameters for DQN:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 1e-4 to 1e-3 | Smaller for stability |
| Batch Size | 32-128 | Balances memory and variance |
| Replay Buffer | 50k-100k | Stores ~500-1000 battles |
| Gamma (γ) | 0.99 | Values long-term outcomes |
| Epsilon Decay | 0.995 | Gradual shift to exploitation |
| Target Update Freq | 100-1000 steps | Stabilizes learning |
| Network Architecture | [256, 256] | Moderate capacity |

### 7.3 Training Challenges

**Challenge #1: Sample Efficiency**
- Battles take 10-50 actions
- Need 1000s of battles to learn basics
- **Solution**: Prioritized replay, curriculum learning

**Challenge #2: Exploration**
- ε-greedy explores uniformly
- Need strategic exploration (try different types)
- **Solution**: Noisy networks, curiosity-driven exploration

**Challenge #3: Opponent Diversity**
- Training vs. one opponent creates overfitting
- **Solution**: Rotate opponents (Random, MaxPower, SimpleHeuristics)

**Challenge #4: Credit Assignment**
- Which action led to victory?
- Early setup or late-game finishing move?
- **Solution**: Multi-step returns, n-step TD learning

---

## 8. Evaluation Metrics

### 8.1 Performance Metrics

**Primary Metric: Win Rate**
```
Win Rate = Wins / Total Battles
```

Evaluated against:
- Random opponent (baseline ~50%)
- MaxBasePowerPlayer (skilled baseline ~30%)
- SimpleHeuristicsPlayer (strong baseline ~20%)

**Secondary Metrics:**

1. **Average Battle Length**: Shorter = more decisive
2. **HP Efficiency**: (Final Team HP - Final Opp HP) / 6
3. **Action Diversity**: Entropy of action distribution
4. **Strategic Indicators**:
   - Setup move usage rate
   - Switch timing (when disadvantaged)
   - Type advantage exploitation

### 8.2 Diagnostic Metrics

**Training Diagnostics:**

1. **Loss Curves**: TD-error should decrease
2. **Q-Value Distribution**: Should grow over time
3. **Reward Distribution**: Should shift positive
4. **Epsilon Decay**: Track exploration→exploitation

**Behavioral Analysis:**

1. **Type Effectiveness Usage**: Does agent use super-effective moves?
2. **Immunity Avoidance**: Does it avoid Electric vs. Ground types?
3. **Hazard Management**: Does it spam Stealth Rock?
4. **Stat Boost Timing**: Does it setup when safe?

---

## 9. Challenges and Solutions

### 9.1 Reward Hacking

**Problem**: Agent exploits reward function loopholes.

**Example**:
```python
# Reward for variety
reward += 0.2 if unique_action_types >= 3 else 0.0

# Exploit: Agent switches randomly for variety bonus
```

**Solution**: Remove explicit incentives; rely on outcome-based learning.

### 9.2 Catastrophic Forgetting

**Problem**: Agent forgets previously learned strategies when learning new ones.

**Solution**:
- Experience replay maintains old experiences
- Curriculum learning (gradual difficulty increase)
- Periodic evaluation on fixed test set

### 9.3 Partial Observability

**Problem**: Agent doesn't know opponent's full team until revealed.

**Solution**:
- Encode "unknown" as zero values
- Use recurrent networks (LSTM) to maintain memory
- Opponent modeling (track revealed moves/abilities)

### 9.4 Stochasticity

**Problem**: Move accuracy, critical hits create randomness.

**Solution**:
- Increase sample size (more battles)
- Use distributional RL (C51, QR-DQN) to model uncertainty
- Reward design should be robust to variance

---

## 10. Future Directions

### 10.1 Algorithmic Improvements

1. **Rainbow DQN**: Combine all DQN improvements
2. **Distributional RL**: Model return distributions (C51, QR-DQN, IQN)
3. **Recurrent Networks**: Use LSTM to handle partial observability
4. **Multi-Agent RL**: Self-play with opponent modeling
5. **Imitation Learning**: Bootstrap from human replays

### 10.2 Domain-Specific Enhancements

1. **Team Building**: Learn to construct competitive teams
2. **Meta-Game Awareness**: Adapt to popular strategies
3. **Opponent Modeling**: Predict opponent's team/actions
4. **Transfer Learning**: Generalize across different tiers/formats
5. **Interpretability**: Explain agent's decision-making

### 10.3 Theoretical Insights

1. **Reward Shaping Theory**: Principled methods for reward design
2. **Exploration in Large Action Spaces**: Better than ε-greedy
3. **Credit Assignment**: Attribute success to specific actions
4. **Sample Efficiency**: Learn from fewer battles

---

## 11. Conclusion

Reinforcement learning for Pokémon Showdown represents a challenging testbed for modern RL algorithms. The domain's strategic depth, partial observability, and stochasticity push the boundaries of current methods.

**Key Takeaways:**

1. **State Representation Matters**: Must encode domain knowledge and strategic context
2. **Reward Design is Critical**: Simple, aligned rewards outperform complex fragile ones
3. **DQN is Sufficient**: Advanced algorithms help but proper design more important
4. **Evaluation Must Be Comprehensive**: Win rate alone insufficient; need behavioral analysis
5. **Iteration is Essential**: V1→V2→V3 shows the iterative refinement process

The journey from naive damage-based rewards to conservative strategic rewards illustrates the core challenges of RL: shaping behavior through incentives while avoiding exploitation. This project demonstrates that successful RL applications require careful consideration of state representation, reward alignment, and algorithmic choices—principles applicable far beyond Pokémon battles.

---

## References

### Foundational RL

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.

### Deep Q-Networks

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *AAAI*.
- Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. *ICML*.
- Schaul, T., et al. (2016). Prioritized experience replay. *ICLR*.
- Fortunato, M., et al. (2018). Noisy networks for exploration. *ICLR*.
- Bellemare, M. G., Dabney, W., & Munos, R. (2017). A distributional perspective on reinforcement learning. *ICML*.
- Hessel, M., et al. (2018). Rainbow: Combining improvements in deep reinforcement learning. *AAAI*.

### Game Playing

- Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484-489.
- Vinyals, O., et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning. *Nature*, 575(7782), 350-354.

### Pokémon AI

- Hu, H., et al. (2024). PokéLLMon: A Human-Parity Agent for Pokémon Battles with Large Language Models. *arXiv preprint arXiv:2402.01118*.

### Reward Shaping

- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*.

---

## Appendix: Network Architecture

```python
class DQNNetwork(nn.Module):
    def __init__(self, state_dim=112, action_dim=9):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
```

**Architecture Choices:**
- **2 hidden layers**: Balances capacity and overfitting
- **256 neurons**: Sufficient for 112-dim state
- **ReLU activation**: Standard for deep networks
- **No dropout**: Experience replay provides regularization
- **Linear output**: Q-values are unbounded

**Alternative Architectures:**
- **Dueling**: Separate V(s) and A(s,a) streams
- **Noisy**: Add parametric noise to weights
- **Recurrent**: LSTM for temporal dependencies
