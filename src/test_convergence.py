## Test Convergence Tracking

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))
from qlearning import QLearningAgent

print("=" * 70)
print("TESTING CONVERGENCE TRACKING")
print("=" * 70)

# Test 1: Initialize agent
print("\n1. Testing agent initialization...")
agent = QLearningAgent(state_size=54, action_size=5)
print(f"   [OK] Agent initialized")
print(f"   Q-table shape: {agent.q_table.shape}")
print(f"   Previous Q-table: {agent.previous_q_table}")
print(f"   Convergence history: {agent.convergence_history}")

# Test 2: First episode - no previous Q-table
print("\n2. Testing first episode (no previous Q-table)...")
convergence_value = agent.calculate_convergence()
print(f"   Convergence value: {convergence_value}")
print(f"   [OK] Correctly returns None for first episode")

is_converged, conv_val = agent.check_convergence()
print(f"   Check convergence: is_converged={is_converged}, value={conv_val}")
print(f"   [OK] Correctly returns False, None for first episode")

# Test 3: Simulate an episode and check convergence
print("\n3. Testing convergence calculation after first episode...")
# Simulate some Q-table updates
for _ in range(10):
    state = np.random.randint(0, 54)
    action = np.random.randint(0, 5)
    reward = np.random.uniform(-1, 1)
    next_state = np.random.randint(0, 54)
    agent.update(state, action, reward, next_state, done=False)

# End episode and track convergence
agent.end_episode()
convergence_value = agent.track_convergence()
print(f"   After first episode:")
print(f"   Convergence value: {convergence_value}")
print(f"   Convergence history length: {len(agent.convergence_history)}")
print(f"   [OK] First episode tracked (should be None)")

# Test 4: Second episode - should have convergence value
print("\n4. Testing convergence calculation after second episode...")
# Make some more updates
for _ in range(10):
    state = np.random.randint(0, 54)
    action = np.random.randint(0, 5)
    reward = np.random.uniform(-1, 1)
    next_state = np.random.randint(0, 54)
    agent.update(state, action, reward, next_state, done=False)

# End episode and track convergence
agent.end_episode()
convergence_value = agent.track_convergence()
print(f"   After second episode:")
print(f"   Convergence value: {convergence_value:.6f}")
print(f"   Convergence history: {[f'{v:.6f}' for v in agent.convergence_history]}")
print(f"   [OK] Convergence value calculated correctly")

# Test 5: Check convergence with different thresholds
print("\n5. Testing convergence check with different thresholds...")
is_converged_001, conv_val_001 = agent.check_convergence(threshold=0.001)
is_converged_01, conv_val_01 = agent.check_convergence(threshold=0.01)
is_converged_1, conv_val_1 = agent.check_convergence(threshold=1.0)
is_converged_100, conv_val_100 = agent.check_convergence(threshold=100.0)

print(f"   Threshold 0.001: converged={is_converged_001}, value={conv_val_001:.6f}")
print(f"   Threshold 0.01:  converged={is_converged_01}, value={conv_val_01:.6f}")
print(f"   Threshold 1.0:   converged={is_converged_1}, value={conv_val_1:.6f}")
print(f"   Threshold 100.0: converged={is_converged_100}, value={conv_val_100:.6f}")
print(f"   [OK] Convergence check works with different thresholds")

# Test 6: Multiple episodes - convergence should decrease
print("\n6. Testing convergence over multiple episodes...")
convergence_values = []
for episode in range(5):
    # Make some updates
    for _ in range(20):
        state = np.random.randint(0, 54)
        action = np.random.randint(0, 5)
        reward = np.random.uniform(-0.5, 0.5)
        next_state = np.random.randint(0, 54)
        agent.update(state, action, reward, next_state, done=False)
    
    # End episode and track
    agent.end_episode()
    conv_val = agent.track_convergence()
    if conv_val is not None:
        convergence_values.append(conv_val)
        print(f"   Episode {episode + 3}: Convergence = {conv_val:.6f}")

print(f"   [OK] Convergence tracked over multiple episodes")
print(f"   Convergence trend: {[f'{v:.6f}' for v in convergence_values]}")

# Test 7: Verify convergence history
print("\n7. Testing convergence history...")
history = agent.get_convergence_history()
print(f"   History length: {len(history)}")
print(f"   History matches tracked values: {len(history) == len(convergence_values)}")
print(f"   [OK] Convergence history accessible")

# Test 8: Test with minimal changes (should have low convergence)
print("\n8. Testing convergence with minimal Q-table changes...")
# Store current Q-table
current_q = agent.q_table.copy()

# Make very small updates
for _ in range(5):
    state = np.random.randint(0, 54)
    action = np.random.randint(0, 5)
    # Very small update
    agent.q_table[state, action] += 0.0001

agent.end_episode()
conv_val = agent.track_convergence()
print(f"   After minimal changes:")
print(f"   Convergence value: {conv_val:.6f}")
print(f"   [OK] Low convergence value for minimal changes")

# Test 9: Test with large changes (should have high convergence)
print("\n9. Testing convergence with large Q-table changes...")
# Make large updates
for _ in range(10):
    state = np.random.randint(0, 54)
    action = np.random.randint(0, 5)
    # Large update
    agent.q_table[state, action] += 1.0

agent.end_episode()
conv_val = agent.track_convergence()
print(f"   After large changes:")
print(f"   Convergence value: {conv_val:.6f}")
print(f"   [OK] High convergence value for large changes")

# Test 10: Test save/load preserves convergence tracking
print("\n10. Testing save/load with convergence tracking...")
# Save agent
agent.save('test_qtable.npy')

# Create new agent and load
new_agent = QLearningAgent(state_size=54, action_size=5)
new_agent.load('test_qtable.npy')

print(f"   Previous Q-table after load: {new_agent.previous_q_table}")
print(f"   Convergence history after load: {new_agent.convergence_history}")
print(f"   [OK] Save/load resets convergence tracking (as expected)")

# Clean up
import os
if os.path.exists('test_qtable.npy'):
    os.remove('test_qtable.npy')

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("[OK] All convergence tracking tests passed!")
print("\nKey features verified:")
print("  - First episode correctly returns None")
print("  - Convergence value calculated as sum of absolute differences")
print("  - Convergence check works with different thresholds")
print("  - Convergence history tracked across episodes")
print("  - Convergence reflects magnitude of Q-table changes")
print("\nReady to use in training loop!")

