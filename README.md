# Pokemon RL Training Setup

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
# Recommended Rainbow configuration
python run.py train cli --gym showdown --domain random --task max Rainbow --episodes 20000 --batch_size 64 --learning_rate 0.00025 --epsilon_decay 0.995 --eval_episodes 400

##OTher training tested
## Training
python run.py train cli --gym showdown --domain random --task max DQN --episodes 10000 --display 1
python run.py train cli --gym showdown --domain random --task max DQN --episodes 5000 --display 1
# Fast training configuration
& ".\venv\pokemon\Scripts\Activate.ps1"
cd gymnasium_envrionments/scripts
python run.py train cli --gym showdown --domain random --task max DQN --episodes 5000 --batch_size 128 --learning_rate 0.0005 --epsilon_decay 0.99 --eval_episodes 1000  --display 1

# Optimized configuration (balanced speed and performance)
python run.py train cli --gym showdown --domain random --task max DQN --episodes 8000 --batch_size 128 --learning_rate 0.0003 --epsilon_decay 0.992 --eval_episodes 400 --display 1
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
# Graceful stop (saves model and logs)
Ctrl + C

# Force stop (emergency only)
Ctrl + Break
```

## Model Persistence & Evaluation

### 📁 **Where Models Are Saved**
Training automatically saves to: `C:\Users\Youjia\cares_rl_logs\Rainbow\Rainbow-random-max-YY_MM_DD_HH-MM-SS\`

**Saved files include:**
- `models/checkpoint/` - Latest model weights
- `alg_config.json` - Algorithm configuration  
- `env_config.json` - Environment configuration
- `train_config.json` - Training configuration
- `training_log.csv` - All training metrics
- `evaluation_log.csv` - Evaluation results

### 🔄 **Resume Training**
```powershell
# Resume from where you left off
python run.py resume cli --data_path "C:\Users\Youjia\cares_rl_logs\Rainbow\Rainbow-random-max-YY_MM_DD_HH-MM-SS"
```

### 🎯 **Evaluate Trained Model**
```powershell
# Evaluate against random opponents (standard evaluation)
python run.py evaluate cli --data_path "C:\Users\Youjia\cares_rl_logs\Rainbow\Rainbow-random-max-YY_MM_DD_HH-MM-SS"

# Test model with custom episodes and seeds
python run.py test cli --data_path "C:\Users\Youjia\cares_rl_logs\Rainbow\Rainbow-random-max-YY_MM_DD_HH-MM-SS" --seeds 42 123 456 --episodes 100
```

### 🏆 **Quick Model Performance Check**
```powershell
# After training, immediately test your model:
& ".\venv\pokemon\Scripts\Activate.ps1"
cd gymnasium_envrionments/scripts

# Find your latest training directory
ls C:\Users\Youjia\cares_rl_logs\Rainbow

# Evaluate it (replace with your actual directory)
python run.py evaluate cli --data_path "C:\Users\Youjia\cares_rl_logs\Rainbow\Rainbow-random-max-25_10_21_15-30-45"
```

### 📊 **Model Comparison**
```powershell
# Test multiple models against each other
python run.py test cli --data_path "path_to_model_1" --seeds 42 --episodes 50
python run.py test cli --data_path "path_to_model_2" --seeds 42 --episodes 50
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
If you encounter KeyError: 'priority' - the showdown_environment.py has been updated with safe attribute access for move properties.

**V2 Network Input**: Make sure observation size matches - now 92 features (was 71 in V1).

Training logs and models are automatically saved to: `cares_rl_logs/DQN/DQN-max-YY_MM_DD_HH-MM-SS/`




