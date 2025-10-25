# Training Optimization Changes - October 25, 2025

## 🎯 What Was Fixed

### ✅ Fix #1: Normalized Reward Scale
**Problem**: Rewards of ±15 caused Q-value explosions and instability  
**Solution**: Normalized to ±1 scale

**Changes**:
```python
# OLD (unstable):
if win: reward += 15.0
if loss: reward -= 10.0
HP progress: ±0.10/turn
Contextual: ±0.35/turn

# NEW (stable):
if win: reward += 1.0
if loss: reward -= 0.3
HP progress: ±0.01/step (scaled by 0.1)
Contextual: ±0.025/step (scaled by 0.05)
```

**Expected episode rewards**: -0.8 to +1.5 (was -20 to +30)

---

### ✅ Fix #2: Dense Per-Step Rewards
**Problem**: Rewards only meaningful at episode end, no temporal credit assignment  
**Solution**: Separated terminal and per-step rewards explicitly

**Changes**:
```python
# NEW structure:
if battle_finished:
    # Terminal rewards only
    if win: reward = 1.0
    elif loss: reward = -0.3
else:
    # Per-step rewards for learning
    reward = hp_progress * 0.1 + contextual * 0.05
```

**Benefits**:
- Agent gets feedback every turn
- Can learn which actions lead to wins
- Dense signal for switching/move timing

---

### ✅ Fix #3: Optimized Hyperparameters
**Problem**: Hyperparameters mismatched to reward scale, causing instability  
**Solution**: Adjusted for normalized rewards and faster learning

| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| **Learning Rate** | 0.0005 | 0.0001 | Matched to ±1 reward scale |
| **Batch Size** | 96 | 32 | Faster adaptation, less memory |
| **Buffer Size** | 50k | 100k | More experience diversity |
| **Epsilon Decay** | 100k steps | 200k steps | More exploration time |
| **End Epsilon** | 0.05 | 0.10 | Continuous exploration |
| **Target Update** | 1000 | 5000 | More stable target network |

**Training Command**:
```powershell
python run.py train cli --gym showdown --domain random --task max Rainbow \
    --batch_size 32 \
    --lr 0.0001 \
    --start_epsilon 1.0 \
    --end_epsilon 0.10 \
    --decay_steps 200000 \
    --number_steps_per_evaluation 2000 \
    --target_update_freq 5000 \
    --buffer_size 100000 \
    --save_train_checkpoints 1
```

---

### ✅ Fix #5: Added Diagnostic Metrics
**Problem**: No visibility into training issues  
**Solution**: Added real-time diagnostics to training logs

**New metrics in logs**:
- `current_step_reward`: Reward for current step (should be in [-1, 1.5])
- `reward_magnitude`: Absolute reward value (should be < 2.0)
- `recent_move_percentage`: % of last 10 actions that were moves
- `recent_switch_percentage`: % of last 10 actions that were switches

**Monitor for red flags**:
- `reward_magnitude > 2.0` → Reward system issue
- `recent_switch_percentage < 0.05` → Agent not exploring switches
- `reward_total` not increasing → Learning stagnation

---

## 📊 Expected Training Behavior

### With Old System (Broken):
```
Episode 0-1000:   Reward: ±10 (random noise)
Episode 1000-5000: Reward: ±8 (still random)
Episode 5000-10000: Reward: ±12 (no improvement)
Win rate: 48-52% (coin flip)
```

### With New System (Fixed):
```
Episode 0-500:    Reward: -0.3 → -0.1 (learning basics)
Episode 500-2000:  Reward: -0.1 → +0.3 (learning strategy)
Episode 2000-5000: Reward: +0.3 → +0.7 (refining tactics)
Episode 5000-10000: Reward: +0.7 → +1.0 (approaching expert)
Win rate: 30% → 40% → 50% → 55%+
```

---

## 🚀 How to Test

### 1. Quick Validation (100 episodes):
```powershell
python run.py train cli --gym showdown --domain random --task max Rainbow \
    --batch_size 32 --lr 0.0001 --start_epsilon 1.0 --end_epsilon 0.10 \
    --decay_steps 200000 --episodes 100 --save_train_checkpoints 1
```

**Check in logs**:
- `reward_magnitude` should be < 2.0
- `current_step_reward` should be in [-1, 1.5]
- No NaN or Inf values

### 2. Short Training Run (1000 episodes):
```powershell
# Same command, but --episodes 1000
```

**Expected after 1000 episodes**:
- Average reward: -0.2 to +0.2 (learning phase)
- Win rate: 35-45% (improving from random)
- Switch percentage: >10% (agent exploring)

### 3. Full Training (10,000+ episodes):
```powershell
# Same command, no episode limit (will run indefinitely)
# Or set --episodes 20000
```

**Expected after 10,000 episodes**:
- Average reward: +0.5 to +0.8
- Win rate: 50-60%
- Clear upward trend in both metrics

---

## 🔍 Debugging Guide

### If rewards are still too large:
**Check**: `reward_magnitude > 2.0` in logs  
**Action**: Verify `calc_reward()` is using normalized scale

### If agent doesn't switch:
**Check**: `recent_switch_percentage < 0.05`  
**Action**: Increase `end_epsilon` to 0.15 for more exploration

### If learning plateaus:
**Check**: Reward stops increasing after 2000 episodes  
**Action**: 
- Reduce learning rate to 0.00005
- Increase buffer size to 200k
- Check opponent difficulty (may need curriculum)

### If Q-values explode:
**Check**: Training logs show loss > 100  
**Action**:
- Verify rewards are normalized
- Reduce learning rate by 50%
- Check for NaN in state representation

---

## 📈 Performance Metrics to Track

Monitor these in your training logs:

| Metric | Target Range | Interpretation |
|--------|-------------|----------------|
| `reward_total` | -0.8 to +1.5 | Episode outcome quality |
| `reward_magnitude` | 0.0 to 2.0 | Per-step reward sanity check |
| `win` | 0.35 to 0.60 | Agent performance vs max |
| `recent_switch_percentage` | 0.10 to 0.30 | Strategic diversity |
| `battle_turns` | 20 to 50 | Game efficiency |
| `team_total_hp` | 2.0 to 5.0 | Win quality |

---

## 🎓 Why These Changes Work

### Mathematical Stability:
**Old system**:
```
Q-update = α × (r + γQ' - Q)
         = 0.0005 × (15 + 0 - 0)
         = 0.0075 per episode
After 100 episodes: Q ≈ 0.75 (unstable)
```

**New system**:
```
Q-update = α × (r + γQ' - Q)
         = 0.0001 × (1.0 + 0 - 0)
         = 0.0001 per episode
After 100 episodes: Q ≈ 0.01 (stable)
```

### Bellman Equation Balance:
```
Target = r + γ·maxQ(s',a')
       = 1.0 + 0.99 × 1.2
       = 2.19

With normalized rewards, target stays bounded!
Old system: target could reach 30+ (chaos)
New system: target stays in [0, 3] (stable)
```

---

## ⚠️ Important Notes

1. **Old models are NOT compatible**: Models trained with ±15 rewards will behave erratically with ±1 rewards. Start fresh.

2. **Curriculum learning still recommended**: While not implemented, training vs random → simple → max opponents would accelerate learning.

3. **Monitor early**: Check logs after 100-500 episodes to catch issues early before wasting computation.

4. **Patience required**: With normalized rewards and proper exploration, you need 5k-10k episodes to see strong results (vs 2k with broken system giving false progress).

---

## 🔄 Rollback Plan

If new system doesn't work, you can revert:

1. Find old commit before October 25 changes
2. Or manually change `calc_reward()` back to:
   ```python
   if win: reward += 15.0
   if loss: reward -= 10.0
   # Remove battle_finished check
   ```
3. Use old hyperparameters from README

But give the new system at least 2,000 episodes before judging!

---

**BOTTOM LINE**: The reward system now provides stable, normalized signals that allow Q-learning to converge properly. Combined with matched hyperparameters, this should produce consistent learning curves rather than random oscillation.
