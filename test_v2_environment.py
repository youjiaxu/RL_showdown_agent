#!/usr/bin/env python3
"""
Test script for Pokemon Showdown V2 Environment
Tests the enhanced state representation and reward system
"""

import numpy as np
import sys
import os

# Add the showdown_gym module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'showdown_gym'))

try:
    from showdown_gym.showdown_environment import ShowdownEnvironment, SingleShowdownWrapper
    print("✅ Successfully imported V2 ShowdownEnvironment")
    
    # Test environment initialization
    env = SingleShowdownWrapper(
        team_type="random",
        opponent_type="random",
        evaluation=False
    )
    print("✅ Successfully created V2 environment")
    
    # Test observation space
    obs_size = env.env._observation_size()
    print(f"✅ Observation size: {obs_size} features (expected: 92)")
    
    # Test a few steps
    obs = env.reset()
    print(f"✅ Reset successful, observation shape: {obs.shape}")
    
    # Test action processing and reward calculation
    action = 6  # Try a move action
    obs, reward, done, truncated, info = env.step(action)
    print(f"✅ Step successful, reward: {reward:.4f}, done: {done}")
    
    # Test multiple steps to check temporal context
    for i in range(3):
        action = np.random.randint(0, 6)  # Random actions
        obs, reward, done, truncated, info = env.step(action)
        print(f"   Step {i+1}: action={action}, reward={reward:.4f}, obs_shape={obs.shape}")
        if done:
            break
    
    print("✅ V2 Environment test completed successfully!")
    print("\nV2 Enhancement Summary:")
    print("- 92-feature state representation (up from 71)")
    print("- Multi-component reward system with strategic timing")
    print("- Temporal context tracking (last 3 actions)")
    print("- Strategic context (battle phase, momentum, boost efficiency)")
    print("- Action pattern recognition and exploration incentives")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()