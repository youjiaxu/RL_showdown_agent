# 🚀 QUICK START - OPTIMIZED TRAINING

## ✅ What Changed (Summary)
1. **Rewards normalized**: ±15 → ±1 scale for stable Q-learning
2. **Learning rate reduced**: 0.0005 → 0.0001 (matched to new reward scale)
3. **Batch size reduced**: 96 → 32 (faster adaptation)
4. **Buffer increased**: 50k → 100k (more experience diversity)
5. **Exploration extended**: 100k → 200k decay steps
6. **Diagnostics added**: Monitor reward magnitude and action distribution

---

## 🎯 Run This Command Now

```powershell
& ".\venv\pokemon\Scripts\Activate.ps1"
cd gymnasium_envrionments/scripts

python run.py train cli --gym showdown --domain random --task max Rainbow --batch_size 32 --lr 0.0001 --start_epsilon 1.0 --end_epsilon 0.10 --decay_steps 200000 --number_steps_per_evaluation 2000 --target_update_freq 5000 --buffer_size 100000 --save_train_checkpoints 1
```

---

## 📊 What to Watch in Training Logs

### ✅ Good Signs:
- `reward_magnitude < 2.0` (rewards normalized correctly)
- `reward_total` increasing over time (-0.3 → 0 → +0.5 → +1.0)
- `recent_switch_percentage > 0.10` (agent exploring switches)
- `win` rate improving (0.30 → 0.40 → 0.50+)

### 🚨 Red Flags:
- `reward_magnitude > 5.0` → Reward system broken
- `reward_total` oscillating wildly → Learning rate too high
- `recent_switch_percentage < 0.05` → Agent not exploring
- `win` rate stuck at 0.48-0.52 → No learning happening

---

## 🕐 Expected Timeline

| Episodes | Expected Reward | Expected Win Rate | What's Happening |
|----------|----------------|-------------------|------------------|
| 0-500 | -0.3 to -0.1 | 30-35% | Learning basics |
| 500-2000 | -0.1 to +0.3 | 35-42% | Discovering strategy |
| 2000-5000 | +0.3 to +0.7 | 42-50% | Refining tactics |
| 5000-10000 | +0.7 to +1.0 | 50-58% | Approaching expert |

---

## 🔍 Quick Debugging

**Problem**: Rewards too large  
**Fix**: Check `calc_reward()` - should return ±1 scale

**Problem**: Not learning  
**Fix**: Reduce lr to 0.00005, check epsilon is decaying properly

**Problem**: Agent doesn't switch  
**Fix**: Increase end_epsilon to 0.15

---

## 📁 Files Modified
- `showdown_environment.py` - Normalized reward system
- `README.md` - Updated training commands
- `OPTIMIZATION_CHANGES_OCT25.md` - Full documentation
- `COMPREHENSIVE_DIAGNOSIS.md` - Problem analysis

---

**Ready to train!** Run the command above and monitor the metrics.
