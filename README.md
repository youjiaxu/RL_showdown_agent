# Pokemon RL Training Setup

## ⚡ V3 STATE ENHANCEMENT: Team-Order Independent Strategic Intelligence

**NEW IN V3**: The environment now provides **125 strategic features** (up from 93) with **TEAM-ORDER INDEPENDENCE** - the critical breakthrough for random Pokemon!

### 🎯 **Type Matchup Intelligence**
- **Team Matchup Matrix**: For each Pokemon, the agent sees:
  - Type effectiveness vs current opponent (0.25x to 4x damage multipliers)
  - How resistant they are to opponent's attacks  
  - Speed advantage/disadvantage
  - Health status and battle readiness
- **Switch Viability Scores**: Direct numerical scores (0-1) for how good each switch would be
- **Coverage Analysis**: Whether moves or switches provide better type coverage

### 🧠 **Professional Player Insights**
- **Threat Assessment**: Offensive/defensive/speed threat levels (what pros evaluate constantly)
- **Speed Tier Analysis**: Critical for competitive play - who goes first matters
- **Hazard Awareness**: Entry hazards and field condition impact
- **Battle Phase Context**: Early/mid/late game strategic shifts

### 🔄 **Why This Fixes Switching**
- **Before**: Agent only saw basic HP/stats, couldn't understand WHY to switch
- **After**: Agent sees exact type advantages (1.5x, 2x, 4x damage), health comparisons, and viability scores
- **Random Pokemon Challenge**: Agent can now evaluate unknown Pokemon based on type matchups rather than memorized strategies

---

## 1. Activate Virtual Environment
```powershell
& ".\venv\pokemon\Scripts\Activate.ps1"
```

## 2. Start Pokemon Showdown Server (in separate terminal)
```powershell
cd pokemon-showdown
node pokemon-showdown start --no-security
```

## 3. Run Training (with venv activated)
```powershell
& ".\venv\pokemon\Scripts\Activate.ps1"
cd gymnasium_envrionments/scripts
#DQN
python run.py train cli --gym showdown --domain random --task max DQN --display 1

#Rainbow DQN
# BALANCED EXPLORATION (Optimized for Move Discovery & Strategic Learning)  
python run.py train cli --gym showdown --domain random --task max Rainbow --batch_size 96 --lr 0.0015 --start_epsilon 1.0 --end_epsilon 0.05 --decay_steps 75000 --number_steps_per_evaluation 2000 --target_update_freq 1000 --buffer_size 50000 --save_train_checkpoints 1

#DQN 
python run.py train cli --gym showdown --domain random --task max DQN --batch_size 96 --lr 0.0015 --start_epsilon 1.0 --end_epsilon 0.08 --decay_steps 100000 --number_steps_per_evaluation 2000 --target_update_freq 1000 --buffer_size 100000 --save_train_checkpoints 1

#PERSAC
python run.py train cli --gym showdown --domain random --task max PERSAC --batch_size 96 --lr 0.001 --start_epsilon 1.0 --end_epsilon 0.05 --decay_steps 75000 --number_steps_per_evaluation 2000 --target_update_freq 1000 --buffer_size 100000 --save_train_checkpoints 1

# ORIGINAL DQN (For Comparison - Assignment allows any algorithm)
python run.py train cli --gym showdown --domain random --task max DQN --episodes 20000 --batch_size 96 --learning_rate 0.0007 --epsilon_start 1.0 --epsilon_decay 0.9999 --epsilon_min 0.1 --number_steps_per_evaluation 2000

## 🧠 Memory Buffer Analysis:

### Small Memory (50k) - BEST for Fast Learning ✅
- **Pro**: Rapid adaptation, fresh experiences, fast initial progress
- **Con**: May forget rare but important scenarios
- **Best for**: Breaking out of performance plateaus, rapid prototyping

### Medium Memory (100k) - RECOMMENDED Balance 🎯  
- **Pro**: Good diversity while maintaining adaptability
- **Con**: Moderate computation overhead
- **Best for**: Stable long-term learning with good performance

### Large Memory (200k) - Most Stable 🛡️
- **Pro**: Remembers diverse scenarios, stable learning
- **Con**: Slower adaptation, higher computation cost
- **Best for**: Final polished training, maximum performance
- **Smaller Memory**: 50k buffer for faster experience turnover
- **Enhanced Rewards**: 3x stronger reward signals for clearer learning

## 🎯 Epsilon Configuration Strategy:

### Improved Epsilon Parameters (NEW)
- **epsilon_decay**: 0.998 (slower decay for better exploration)
- **epsilon_min**: 0.05 (maintains 5% exploration permanently)
- **Benefits**: Better exploration-exploitation balance, prevents complete exploitation phase

### 🚨 CRITICAL EPSILON TROUBLESHOOTING:

#### **Problem: epsilon: 0.0 from Episode 1**
**Symptoms**: Training logs show `epsilon: 0.0` immediately from first episode
**Root Cause**: Missing `--epsilon_start 1.0` parameter in training command
**Solution**: ALWAYS include `--epsilon_start 1.0` in your training command

#### **Problem: epsilon: 0.0 after ~100-200 episodes**
**Symptoms**: Epsilon starts at 1.0 but hits 0.0 too quickly (episodes 100-200)
**Root Cause**: epsilon_decay parameter too aggressive (like 0.99, 0.995, 0.997)
**Solution**: Use ultra-conservative epsilon_decay (0.99995 or 0.99999)

#### **Mathematical Epsilon Analysis:**
```
With epsilon_start=1.0, epsilon_decay=0.99995, epsilon_min=0.12:
Episode 1:    epsilon = 1.000 (100% exploration)
Episode 100:  epsilon = 0.995 (99.5% exploration)  
Episode 500:  epsilon = 0.975 (97.5% exploration)
Episode 1000: epsilon = 0.951 (95.1% exploration)
Episode 2000: epsilon = 0.905 (90.5% exploration)
Episode 5000: epsilon = max(0.779, 0.12) = 0.779 (77.9% exploration)
Episode 10000: epsilon = max(0.607, 0.12) = 0.607 (60.7% exploration)
```

#### **Why Pokemon Needs Ultra-Conservative Epsilon:**
- **Strategic Complexity**: 18 types × 18 types = 324 type interactions to discover
- **Team Composition**: 6 Pokemon × 4 moves each = 24 actions per Pokemon
- **Switching Timing**: Critical to learn when to switch vs when to attack
- **Setup Strategies**: Stat boosts, weather, terrain effects require exploration
- **Pure Exploitation (epsilon=0.0)**: Agent never discovers new strategies → poor performance

## 🎯 Showdown Gym Command Structure:

### **Correct Command Syntax:**
```powershell
python run.py train cli --gym showdown --domain random --task max [ALGORITHM] [PARAMETERS]
```

### **Parameter Explanation:**
- **`train`**: Training mode (vs evaluate, test, resume)
- **`cli`**: Command-line interface for showdown gym (REQUIRED for this environment)
- **`--gym showdown`**: Use Pokemon Showdown environment 
- **`--domain random`**: Random team generation (required for assignment)
- **`--task max`**: Train against "max agent" expert opponent (assignment target)
- **`[ALGORITHM]`**: Your agent's learning algorithm (`Rainbow`, `DQN`, etc.)
- **`--epsilon_start 1.0`**: **CRITICAL** - Initialize epsilon to 1.0 (was missing!)

### **Assignment Context:**
- **Goal**: Beat the "max agent" in random domain battles
- **Algorithm Choice**: Assignment allows any algorithm (Rainbow recommended over DQN)
- **Domain**: `random` creates random teams each game (challenging but required)
- **Task**: `max` is your target opponent (expert agent you need to beat)
```

## Quick Start (all-in-one command)
```powershell
& ".\venv\pokemon\Scripts\Activate.ps1"; cd gymnasium_envrionments/scripts; python run.py train cli --gym showdown --domain random --task max DQN --display 1
```

## Plotting
```powershell
cd cares_reinforcement_learning/cares_reinforcement_learning/util
# List available training runs
ls C:\Users\Youjia\cares_rl_logs\DQN

# Plot results (replace with your actual training directory)
python plotter.py -s C:\Users\Youjia\cares_rl_logs -d C:\Users\Youjia\cares_rl_logs\DQN\DQN-random-max-YY_MM_DD_HH-MM-SS --y_train win --y_eval win

# Example with actual directory:
python plotter.py -s C:\Users\Youjia\cares_rl_logs -d C:\Users\Youjia\cares_rl_logs\DQN\DQN-random-max-25_10_18_16-32-26 --y_train win --y_eval win

# Plots are saved to: C:\Users\Youjia\cares_rl_logs\figures\
# - random-max-compare-eval.png (evaluation results)
# - random-max-compare-train.png (training results)
```

## Stopping Training
```powershell
# After stopping, verify what was saved:
ls "C:\Users\Youjia\cares_rl_logs\Rainbow\Rainbow-random-max-YY_MM_DD_HH-MM-SS\10\models"
ls "C:\Users\Youjia\cares_rl_logs\Rainbow\Rainbow-random-max-YY_MM_DD_HH-MM-SS\10\memory"

# If memory directory is empty or has old timestamp:
# Memory buffer was not saved - you'll need to retrain from last checkpoint
```

## Model Persistence & Evaluation

### 📁 **Where Models Are Saved**
Training automatically saves to: `C:\Users\Youjia\cares_rl_logs\Rainbow\Rainbow-random-max-YY_MM_DD_HH-MM-SS\`

**Saved files include:**
- `models/checkpoint/` - Latest model weights
- `alg_config.json` - Algorithm configuration  
- `env_config.json` - Environment configuration
- `train_config.json` - Training configuration
- `training_log.csv` - All training metrics (updated every episode)
- `evaluation_log.csv` - Evaluation results (updated every checkpoint)

### 🔄 **Resume Training from Checkpoint**
```powershell
# Resume from where you left off (uses SAME hyperparameters from saved config)
# CORRECT SYNTAX: No "cli" for resume command
python run.py resume --data_path "C:\Users\Youjia\cares_rl_logs\Rainbow\Rainbow-random-max-YY_MM_DD_HH-MM-SS" --seed 42
```

**⚠️ IMPORTANT: Resume Behavior**
- **Hyperparameters**: Uses the SAME hyperparameters from saved `alg_config.json`, `env_config.json`, `train_config.json`
- **Episode Limit**: Continues until reaching the ORIGINAL episode limit from saved config
- **Memory Buffer**: Loads the exact experience replay buffer state
- **Model Weights**: Loads the exact network weights from checkpoint

### 🚀 **Extending Training Beyond Episode Limit**

**Problem**: If your model reaches the original episode limit (e.g., 20,000), resume won't train further.

**Solution 1: Manual Config Edit**
```powershell
# Navigate to your saved model directory
cd "C:\Users\Youjia\cares_rl_logs\Rainbow\Rainbow-random-max-YY_MM_DD_HH-MM-SS"

# Then resume training
python run.py resume --data_path "." --seed 42
```

**Solution 2: Fresh Training with Loaded Weights (Advanced)**
```powershell
# Start new training session but manually load the weights
# This gives you full control over new hyperparameters and episode limits
python run.py train cli --gym showdown --domain random --task max Rainbow --episodes 40000 --batch_size 64 --learning_rate 0.00025 --epsilon_decay 0.995 --eval_episodes 400
# Then manually load weights in code (requires modification)
```

**Solution 3: Multiple Resume Sessions**
```powershell
# Increase episodes incrementally
# Edit train_config.json: episodes 20000 → 25000
python run.py resume --data_path "your_path" --seed 42

# After completion, edit again: 25000 → 30000  
python run.py resume --data_path "your_path" --seed 42
```

### 🎯 **Evaluate Trained Model**
```powershell
# Evaluate against random opponents (standard evaluation)
# CORRECT SYNTAX: No "cli" for evaluate command
python run.py evaluate --data_path "C:\Users\Youjia\cares_rl_logs\Rainbow\Rainbow-random-max-YY_MM_DD_HH-MM-SS"

# Test model with custom episodes and seeds
# CORRECT SYNTAX: No "cli" for test command
python run.py test --data_path "C:\Users\Youjia\cares_rl_logs\Rainbow\Rainbow-random-max-YY_MM_DD_HH-MM-SS" --seeds 42 123 456 --episodes 100
```

### 🏆 **Quick Model Performance Check**
```powershell
# After training, immediately test your model:
& ".\venv\pokemon\Scripts\Activate.ps1"
cd gymnasium_envrionments/scripts

# Find your latest training directory
ls C:\Users\Youjia\cares_rl_logs\Rainbow

# Evaluate it (replace with your actual directory)
python run.py evaluate --data_path "C:\Users\Youjia\cares_rl_logs\Rainbow\Rainbow-random-max-25_10_21_15-30-45"
```

### 📊 **Model Comparison**
```powershell
# Test multiple models against each other
python run.py test --data_path "path_to_model_1" --seeds 42 --episodes 50
python run.py test --data_path "path_to_model_2" --seeds 42 --episodes 50
# Same seed ensures fair comparison
```

## V2 Enhanced Features

### 🧠 Strategic Learning Improvements
- **Enhanced State Representation**: 92 features (up from 71) with temporal and strategic context
- **Multi-Component Rewards**: Strategic timing rewards, efficiency bonuses, exploration incentives
- **Temporal Context**: Tracks last 3 actions and their outcomes for better decision making
- **Battle Phase Awareness**: Early/mid/late game indicators for context-appropriate strategies
- **Momentum Tracking**: Recent damage trends to understand battle flow

### 🎯 Key V2 Enhancements
1. **Stat Boost Intelligence**: Penalties for redundant boosts, rewards for good timing
2. **Action Pattern Recognition**: Detects and rewards strategic variety vs repetitive play
3. **Efficiency Rewards**: Bonuses for winning quickly, penalties for overly long battles
4. **Strategic Context**: Battle phase, boost effectiveness, momentum indicators

### 🚀 Expected Improvements
- Reduced stat boost spam through strategic timing rewards
- Better strategic decision-making with temporal context
- Smoother learning curves through enhanced reward shaping
- More sophisticated gameplay through multi-component objectives

## Troubleshooting

### 🐛 **Common Issues & Solutions**

#### **AssertionError: Error with move [movename] (poke-env 0.10.0 Compatibility)**
```
AssertionError: Error with move focusblast. Expected self.moves to contain copycat, metronome, mefirst, mirrormove, assist, transform or mimic.
```

**Cause**: You're using poke-env 0.10.0 (as specified in requirements.txt) which has stricter move validation. This version sometimes fails when Pokemon Showdown sends unexpected move combinations, especially after connection timeouts or server idle periods.

**Root Cause**: Pokemon Showdown server times out every ~15 minutes of inactivity, causing state desynchronization with poke-env.

**Prevention Strategies (in order of effectiveness)**:

1. **Proactive Server Restart Every 1000-2000 Episodes** (Best Prevention):
   ```powershell
   # During training, periodically restart Pokemon Showdown server:
   # Stop training gracefully (Ctrl+C)
   # In showdown terminal: Ctrl+C
   # Restart: node pokemon-showdown start --no-security
   # Resume training normally
   ```

2. **Monitor Training Progress and Restart at Signs of Issues**:
   ```powershell
   # Watch for these warning signs:
   # - Episodes taking unusually long (>10 seconds each)
   # - Websocket connection warnings in terminal
   # - Repeated "waiting for battle" messages
   # - Any poke-env parsing errors
   
   # When you see these, restart showdown server immediately
   ```

3. **Use Smaller Episode Batches for Stability**:
   ```powershell
   # Instead of 20,000 episodes in one go, use smaller batches:
   python run.py train cli --gym showdown --domain random --task max Rainbow --episodes 5000 --batch_size 64 --learning_rate 0.00025 --epsilon_decay 0.995 --eval_episodes 400
   
   # Then resume/continue:
   # Edit train_config.json to increase episodes and resume
   ```

4. **Immediate Recovery When Error Occurs**:
   ```powershell
   # If you see the assertion error during training:
   
   # Step 1: Stop training (Ctrl+C)
   # Step 2: Restart showdown server
   cd pokemon-showdown
   Ctrl+C
   node pokemon-showdown start --no-security
   
   # Step 3: Resume training (your progress is saved)
   python run.py resume --data_path "your_saved_model_path" --seed 42
   ```

**Why Pokemon Showdown Timeouts Cause This**:
- Server goes idle after 15 minutes → stale battle states
- poke-env expects consistent move/Pokemon state
- Timeout creates inconsistent moveset data
- Assertion fails when moves don't match expected patterns

**Best Practice for Long Training Sessions**:
```powershell
# With the connection fix, you should be able to run much longer:
# 1. Try full 20,000 episodes without interruption first
# 2. If any issues occur, restart server every 5,000 episodes  
# 3. The connection timeout fix should handle most stability issues
```

**Legacy Solutions (if prevention fails)**:

1. **Restart Pokemon Showdown Server** (Most Effective):
   ```powershell
   # In the showdown terminal, stop server:
   Ctrl + C
   
   # Restart server:
   cd pokemon-showdown
   node pokemon-showdown start --no-security
   
   # Then restart training - this resolves 90% of cases
   ```

2. **Use Gen 8 Random Battles** (More Stable):
   ```powershell
   # Gen 8 has better compatibility with poke-env 0.10.0
   python run.py train cli --gym showdown --domain random --task max Rainbow --episodes 20000 --batch_size 64 --learning_rate 0.00025 --epsilon_decay 0.995 --eval_episodes 400
   # Edit battle format in environment if needed
   ```

3. **Temporary Skip Strategy**:
   ```powershell
   # If error occurs during training, just restart
   # Your progress is automatically saved every few episodes
   # Resume from last checkpoint:
   python run.py resume --data_path "your_saved_model_path" --seed 42
   ```

**Why We Keep poke-env 0.10.0**:
- Project specifically uses this version for stability
- Newer versions (0.13.x) may have breaking changes
- Your environment is designed and tested with 0.10.0

**Important**: DO NOT upgrade poke-env unless specifically instructed. The project dependencies are locked for compatibility.

---

## 🎯 V3 Enhanced Features: Understanding the Strategic Intelligence

### **What Changed in the State Representation**

**Before V3 (93 features)**:
- Basic HP, stats, moves, weather
- Limited temporal context (3 recent actions)
- No explicit type matchup information
- Agent had to "guess" switching value

**After V3 (138 features)**:
- **30 Type Matchup Features**: For each of your 6 Pokemon vs the current opponent
  - Offensive type effectiveness (0.25x, 0.5x, 1x, 2x, 4x damage multipliers)
  - Defensive type effectiveness (how much damage they can deal to you)
  - Speed comparison (who goes first - critical for switching)
  - Health status and battle readiness score
  - Overall viability rating for that switch

- **6 Switch Viability Scores**: Direct 0-1 ratings telling the agent exactly how good each switch would be right now

- **3 Threat Assessment Values**: What professional players constantly evaluate
  - Offensive threat: "Can I KO the opponent?"
  - Defensive threat: "Can they KO me?"
  - Speed threat: "Who controls the tempo?"

- **Professional Insights**: Speed tiers, coverage analysis, hazard awareness, field conditions

### **Why This Should Fix Switching**

**The Core Problem**: 
- Pokemon agents avoided switching because they couldn't see the strategic value
- With random Pokemon, agents can't memorize matchups - they need to understand types
- Type effectiveness is THE fundamental strategic layer in Pokemon

**The V3 Solution**:
1. **Explicit Type Information**: Agent now sees "Switching to Pokemon #3 gives me 2x damage advantage"
2. **Risk Assessment**: Agent knows "Current Pokemon has 0.3 HP, switch target has 0.9 HP"  
3. **Speed Control**: Agent understands "Switch target is faster, can revenge kill"
4. **Opportunity Cost**: Agent can compare "Stay and deal 0.5x damage vs switch for 2x damage"

### **Expected Training Behavior Changes**

**Early Training (Episodes 1-5000)**:
- Increased switching attempts as agent discovers type advantages
- switch_percentage should rise from 0% to 15-25%
- More defensive switches when HP is low

**Mid Training (Episodes 5000-15000)**:
- Strategic switching for type advantages
- Better timing (switching when safe, not when critical)
- Understanding speed control and revenge killing

**Late Training (Episodes 15000+)**:
- Optimal switching patterns
- Complex reads (switching to counter predicted switches)
- Master-level type matchup exploitation

### **Monitoring Your Training**

**Key Metrics to Watch**:
```
switch_percentage: Should increase from ~0% to 20-35%
good_defensive_switch: Should increase (saving low HP Pokemon)
reward_total: Should be higher due to better type matchup utilization
victory_efficiency: Should improve (faster wins through smart switching)
```

**If Switching Still Doesn't Happen**:
1. **Use SWITCHING DISCOVERY config** (20% end_epsilon with 150k decay_steps)
2. **Check switch_percentage in logs** - should be >10% by episode 2000
3. **Verify type effectiveness calculation** - agent should see clear advantages

**Random Pokemon Advantage**:
- Agent can't memorize specific Pokemon matchups
- Must learn general type interaction principles  
- Forces true strategic understanding rather than pattern memorization

#### **KeyError: 'priority'**
Environment has been updated with safe attribute access for move properties in poke-env 0.10.0.

#### **Connection Issues**
```
ConnectionError: Could not connect to Pokemon Showdown server
```

**Solution**: Ensure Pokemon Showdown server is running on correct port:
```powershell
cd pokemon-showdown
node pokemon-showdown start --no-security
# Should show: "Server started on port 8000"
```

#### **V2 Network Input Size**
**Current**: 93 features (updated from original 71 in V1)
**Poke-env 0.10.0**: Compatible with current feature extraction

#### **Training Hangs or Memory Issues**
For poke-env 0.10.0 stability:
- Use smaller batch sizes (32 instead of 64) if memory issues occur
- Restart showdown server every 1000-2000 episodes for best stability
- Monitor both terminals for any websocket disconnection messages

### 📦 **Version Compatibility**
```
poke-env: 0.10.0 (locked for project compatibility)
numpy: 2.2.5
Python: 3.10+
Pokemon Showdown: Latest (compatible with poke-env 0.10.0)
```

### 📊 **Performance Monitoring**
Training logs and models are automatically saved to: `C:\Users\Youjia\cares_rl_logs\Rainbow\Rainbow-random-max-YY_MM_DD_HH-MM-SS\`

**Monitor training progress**:
- Watch terminal output for episode wins/losses
- Check evaluation results every few hundred episodes
- Look for increasing win rates over time
- poke-env 0.10.0 provides stable performance metrics




