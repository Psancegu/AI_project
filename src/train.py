"""
End-to-end training script for Q-Learning trading agent.

This script demonstrates the complete workflow:
1. Load and split data
2. Initialize environment and agent
3. Train the agent
4. Track convergence
5. Test on unseen data
6. Evaluate performance
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(__file__))

from environment import TradingEnvironment
from qlearning import QLearningAgent
from utils import (
    load_data,
    split_data,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    plot_portfolio_performance,
)


def train_agent(env, agent, num_episodes=1000, convergence_threshold=0.01, 
                print_interval=100):
    """
    Train the Q-learning agent.
    
    Args:
        env: Trading environment
        agent: Q-learning agent
        num_episodes: Number of training episodes
        convergence_threshold: Convergence threshold for early stopping
        print_interval: Print progress every N episodes
        
    Returns:
        Dictionary with training metrics
    """
    print(f"\n{'='*70}")
    print(f"TRAINING AGENT")
    print(f"{'='*70}")
    print(f"Episodes: {num_episodes}")
    print(f"Convergence threshold: {convergence_threshold}")
    print(f"{'='*70}\n")
    
    episode_rewards = []
    episode_portfolio_values = []
    convergence_values = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
        
        # End episode and track convergence
        agent.end_episode()
        conv_value = agent.track_convergence()
        
        # Store metrics
        final_portfolio_value = info['portfolio_value']
        episode_rewards.append(episode_reward)
        episode_portfolio_values.append(final_portfolio_value)
        if conv_value is not None:
            convergence_values.append(conv_value)
        
        # Print progress
        if (episode + 1) % print_interval == 0 or episode == 0:
            conv_str = f"{conv_value:.6f}" if conv_value is not None else "N/A"
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Portfolio: ${final_portfolio_value:10.2f} | "
                  f"Convergence: {conv_str}")
        
        # Check for convergence
        if conv_value is not None:
            is_converged, _ = agent.check_convergence(threshold=convergence_threshold)
            if is_converged:
                print(f"\n[CONVERGED] at episode {episode + 1} with convergence: {conv_value:.6f}")
                break
    
    return {
        'episode_rewards': episode_rewards,
        'episode_portfolio_values': episode_portfolio_values,
        'convergence_values': convergence_values,
        'num_episodes': len(episode_rewards)
    }


def test_agent(env, agent):
    """
    Test the trained agent on unseen data.
    
    Args:
        env: Trading environment (test data)
        agent: Trained Q-learning agent
        
    Returns:
        Dictionary with test metrics
    """
    print(f"\n{'='*70}")
    print(f"TESTING AGENT")
    print(f"{'='*70}\n")
    
    state = env.reset()
    done = False
    portfolio_values = [env.initial_balance]
    actions_taken = []
    rewards = []
    
    step = 0
    while not done:
        action = agent.act(state, training=False)  # No exploration in testing
        next_state, reward, done, info = env.step(action)
        
        state = next_state
        portfolio_values.append(info['portfolio_value'])
        actions_taken.append(action)
        rewards.append(reward)
        step += 1
        
        if step % 100 == 0:
            print(f"Step {step:4d} | Portfolio: ${info['portfolio_value']:10.2f} | "
                  f"Cash: ${info['cash']:10.2f} | Holdings: {len(info['holdings'])} stocks")
    
    # Calculate metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - env.initial_balance) / env.initial_balance * 100
    
    portfolio_series = pd.Series(portfolio_values)
    returns = portfolio_series.pct_change().dropna()
    
    sharpe = calculate_sharpe_ratio(returns) if len(returns) > 0 else 0.0
    max_dd = calculate_max_drawdown(portfolio_values)
    
    # Action distribution
    action_names = ['Hold', 'Buy Conservative', 'Buy Aggressive', 
                    'Sell Conservative', 'Sell Aggressive']
    action_counts = pd.Series(actions_taken).value_counts().sort_index()
    
    print(f"\n{'='*70}")
    print(f"TEST RESULTS")
    print(f"{'='*70}")
    print(f"Initial Balance: ${env.initial_balance:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"\nAction Distribution:")
    for action_idx, count in action_counts.items():
        print(f"  {action_names[action_idx]}: {count} times ({count/len(actions_taken)*100:.1f}%)")
    
    return {
        'portfolio_values': portfolio_values,
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'actions_taken': actions_taken,
        'rewards': rewards
    }


def plot_training_results(train_metrics, test_metrics):
    """
    Plot training and testing results.
    
    Args:
        train_metrics: Dictionary with training metrics
        test_metrics: Dictionary with test metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training portfolio values
    axes[0, 0].plot(train_metrics['episode_portfolio_values'])
    axes[0, 0].set_title('Training: Portfolio Value per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training rewards
    axes[0, 1].plot(train_metrics['episode_rewards'])
    axes[0, 1].set_title('Training: Reward per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Convergence
    if len(train_metrics['convergence_values']) > 0:
        axes[1, 0].plot(train_metrics['convergence_values'])
        axes[1, 0].set_title('Training: Convergence per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Convergence (Sum of Abs Differences)')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Test portfolio performance
    axes[1, 1].plot(test_metrics['portfolio_values'])
    axes[1, 1].axhline(y=test_metrics['portfolio_values'][0], 
                       color='r', linestyle='--', label='Initial Balance')
    axes[1, 1].set_title('Test: Portfolio Value Over Time')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Portfolio Value ($)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] Training results plot to 'training_results.png'")
    plt.show()


def main():
    """
    Main training script.
    """
    print("=" * 70)
    print("Q-LEARNING TRADING AGENT - TRAINING SCRIPT")
    print("=" * 70)
    
    # Configuration
    DATA_PATH = 'data/constituents.csv'
    INITIAL_BALANCE = 10000
    NUM_EPISODES = 1000
    CONVERGENCE_THRESHOLD = 0.01
    LEARNING_RATE = 0.01
    DISCOUNT_FACTOR = 0.95
    EPSILON = 1.0
    EPSILON_DECAY = 0.995
    EPSILON_MIN = 0.01
    
    # Step 1: Load data
    print(f"\n[1/6] Loading data from {DATA_PATH}...")
    try:
        if DATA_PATH.endswith('.parquet'):
            try:
                data = pd.read_parquet(DATA_PATH)
            except ImportError:
                print(f"   [WARNING] Parquet support not available. Installing pyarrow...")
                print(f"   Please run: pip install pyarrow")
                print(f"   Or use CSV format instead.")
                raise ImportError("pyarrow is required for parquet files. Install with: pip install pyarrow")
        elif DATA_PATH.endswith('.csv'):
            data = load_data(DATA_PATH)
        else:
            raise ValueError(f"Unsupported file format: {DATA_PATH}")
        
        print(f"   [OK] Loaded {len(data)} rows")
        print(f"   Date range: {data.index.min()} to {data.index.max()}")
        print(f"   Stocks: {sorted(data['Ticker'].unique()) if 'Ticker' in data.columns else 'N/A'}")
    except FileNotFoundError:
        print(f"   [ERROR] Data file not found: {DATA_PATH}")
        print(f"   Please update DATA_PATH in the script or create synthetic data for testing.")
        print(f"   You can use train_example.py for testing with synthetic data.")
        return
    except ImportError as e:
        print(f"   [ERROR] {e}")
        print(f"   To install pyarrow, run: pip install pyarrow")
        return
    except Exception as e:
        print(f"   [ERROR] Failed to load data: {e}")
        print(f"   You can use train_example.py for testing with synthetic data.")
        return
    
    # Step 2: Split data
    print(f"\n[2/6] Splitting data into training and test sets...")
    try:
        train_data, test_data = split_data(data)
        print(f"   [OK] Training data: {len(train_data)} rows")
        print(f"   [OK] Test data: {len(test_data)} rows")
    except Exception as e:
        print(f"   [ERROR] Failed to split data: {e}")
        return
    
    # Step 3: Initialize environment and agent
    print(f"\n[3/6] Initializing environment and agent...")
    try:
        train_env = TradingEnvironment(train_data, initial_balance=INITIAL_BALANCE)
        test_env = TradingEnvironment(test_data, initial_balance=INITIAL_BALANCE)
        
        agent = QLearningAgent(
            state_size=54,
            action_size=5,
            learning_rate=LEARNING_RATE,
            discount_factor=DISCOUNT_FACTOR,
            epsilon=EPSILON,
            epsilon_decay=EPSILON_DECAY,
            epsilon_min=EPSILON_MIN
        )
        
        print(f"   [OK] Training environment initialized")
        print(f"   [OK] Test environment initialized")
        print(f"   [OK] Q-learning agent initialized (Q-table shape: {agent.q_table.shape})")
    except Exception as e:
        print(f"   [ERROR] Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Train agent
    print(f"\n[4/6] Training agent...")
    try:
        train_metrics = train_agent(
            train_env, 
            agent, 
            num_episodes=NUM_EPISODES,
            convergence_threshold=CONVERGENCE_THRESHOLD
        )
        print(f"   [OK] Training completed")
    except Exception as e:
        print(f"   [ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Test agent
    print(f"\n[5/6] Testing agent on unseen data...")
    try:
        test_metrics = test_agent(test_env, agent)
        print(f"   [OK] Testing completed")
    except Exception as e:
        print(f"   [ERROR] Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Save results
    print(f"\n[6/6] Saving results...")
    try:
        # Save Q-table
        agent.save('q_table_trained.npy')
        print(f"   [OK] Q-table saved to 'q_table_trained.npy'")
        
        # Plot results
        plot_training_results(train_metrics, test_metrics)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'episode': range(1, len(train_metrics['episode_portfolio_values']) + 1),
            'portfolio_value': train_metrics['episode_portfolio_values'],
            'reward': train_metrics['episode_rewards'],
            'convergence': train_metrics['convergence_values'] + [None] * 
                          (len(train_metrics['episode_portfolio_values']) - 
                           len(train_metrics['convergence_values']))
        })
        metrics_df.to_csv('training_metrics.csv', index=False)
        print(f"   [OK] Training metrics saved to 'training_metrics.csv'")
        
    except Exception as e:
        print(f"   [WARNING] Failed to save some results: {e}")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nSummary:")
    print(f"  Training episodes: {train_metrics['num_episodes']}")
    print(f"  Final training portfolio: ${train_metrics['episode_portfolio_values'][-1]:,.2f}")
    print(f"  Test portfolio return: {test_metrics['total_return']:.2f}%")
    print(f"  Test Sharpe ratio: {test_metrics['sharpe_ratio']:.4f}")
    print(f"  Test max drawdown: {test_metrics['max_drawdown']:.2f}%")


if __name__ == '__main__':
    main()

