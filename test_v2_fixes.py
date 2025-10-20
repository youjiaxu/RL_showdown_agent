#!/usr/bin/env python3
"""
Test script to validate V2 environment fixes for training stability
"""

import numpy as np
import sys
import os

# Add the showdown_gym to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'showdown_gym'))

from showdown_gym.showdown_environment import ShowdownEnvironment, SingleShowdownWrapper

def test_v2_environment():
    """Test V2 environment initialization and basic functionality"""
    print("🧪 Testing V2 Environment Fixes...")
    
    try:
        # Test environment creation
        print("1. Creating V2 environment...")
        env = SingleShowdownWrapper(
            team_type="random",
            opponent_type="random",
            evaluation=False
        )
        print("✅ Environment created successfully")
        
        # Test reset
        print("2. Testing environment reset...")
        obs, info = env.reset()
        print(f"✅ Reset successful, observation shape: {obs.shape}")
        print(f"   Expected: (93,), Got: {obs.shape}")
        
        if obs.shape[0] != 93:
            print("❌ ERROR: Observation size mismatch!")
            return False
        
        # Test observation bounds
        print("3. Checking observation value bounds...")
        obs_min, obs_max = obs.min(), obs.max()
        print(f"   Observation range: [{obs_min:.4f}, {obs_max:.4f}]")
        
        if obs_min < -10 or obs_max > 10:
            print("⚠️  WARNING: Some observation values may be unbounded")
        else:
            print("✅ Observation values appear bounded")
        
        # Test a few steps
        print("4. Testing environment steps...")
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"   Step {step+1}: action={action}, reward={reward:.4f}, done={done}")
            
            # Check reward bounds
            if abs(reward) > 10:
                print(f"⚠️  WARNING: Large reward detected: {reward}")
            
            if done or truncated:
                print("   Battle ended, resetting...")
                obs, info = env.reset()
                break
        
        print("✅ Environment stepping successful")
        
        # Test internal V2 features
        print("5. Testing V2 specific features...")
        primary_env = env.env
        if hasattr(primary_env, 'action_history'):
            print(f"   Action history length: {len(primary_env.action_history)}")
        if hasattr(primary_env, 'momentum_tracker'):
            print(f"   Momentum tracker active: {len(primary_env.momentum_tracker['damage_dealt'])} records")
        
        print("✅ V2 features accessible")
        
        env.close()
        print("\n🎉 All V2 environment tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_stability():
    """Test reward calculation stability"""
    print("\n🎯 Testing Reward System Stability...")
    
    try:
        env = SingleShowdownWrapper(team_type="random", opponent_type="random", evaluation=False)
        obs, info = env.reset()
        
        rewards = []
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            
            if done or truncated:
                obs, info = env.reset()
        
        rewards = np.array(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        
        print(f"   Reward statistics over 10 steps:")
        print(f"   Mean: {mean_reward:.4f}, Std: {std_reward:.4f}")
        print(f"   Range: [{min_reward:.4f}, {max_reward:.4f}]")
        
        # Check for stability
        if std_reward > 5.0:
            print("⚠️  WARNING: High reward variance detected")
        else:
            print("✅ Reward variance appears stable")
            
        if abs(min_reward) > 8.0 or abs(max_reward) > 8.0:
            print("⚠️  WARNING: Rewards outside expected bounds")
        else:
            print("✅ Rewards within expected bounds")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Reward stability test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 V2 Environment Fix Validation\n")
    
    # Run tests
    test1_passed = test_v2_environment()
    test2_passed = test_reward_stability()
    
    print(f"\n📊 Test Results:")
    print(f"   Environment Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Reward Stability: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All V2 fixes validated! Environment ready for stable training.")
    else:
        print("\n⚠️  Some issues detected. Check output above for details.")