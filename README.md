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
cd gymnasium_envrionments/scripts
python run.py train cli --gym showdown --domain random --task max DQN --display 1
```

## Quick Start (all-in-one command)
```powershell
& ".\venv\pokemon\Scripts\Activate.ps1"; cd gymnasium_envrionments/scripts; python run.py train cli --gym showdown --domain random --task max DQN --display 1
```
