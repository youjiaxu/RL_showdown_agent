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
python run.py train cli --gym showdown --domain random --task max DQN --display 1
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

## Troubleshooting
If you encounter KeyError: 'priority' - the showdown_environment.py has been updated with safe attribute access for move properties.

Training logs and models are automatically saved to: `cares_rl_logs/DQN/DQN-max-YY_MM_DD_HH-MM-SS/`




