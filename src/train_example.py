"""
Example training script with synthetic data for testing.

This is a simplified version that creates synthetic data for testing
the complete training pipeline.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.dirname(__file__))

from environment import TradingEnvironment
from qlearning import QLearningAgent
from utils import split_data

# Create synthetic data
print("Creating synthetic data for testing...")
np.random.seed(42)

# Create date range (2000-2024)
dates = pd.date_range(start='2000-01-01', end='2024-12-31', freq='D')
dates = dates[dates.weekday < 5]  # Only weekdays

# Create 10 stocks + index
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT', '^GSPC']
n_days = len(dates)

data_rows = []
for ticker in tickers:
    base_price = np.random.uniform(50, 200) if ticker != '^GSPC' else 100
    trend = np.linspace(0, np.random.uniform(-30, 50), n_days)
    noise = np.random.randn(n_days).cumsum() * np.random.uniform(1, 3)
    prices = base_price + trend + noise
    prices = np.maximum(prices, 1)
    
    for i, date in enumerate(dates):
        data_rows.append({
            'Date': date,
            'Ticker': ticker,
            'Open': prices[i] * 0.99,
            'High': prices[i] * 1.02,
            'Low': prices[i] * 0.98,
            'Close': prices[i],
            'Volume': np.random.randint(1000000, 10000000)
        })

data = pd.DataFrame(data_rows)
data = data.set_index('Date')

print(f"Created {len(data)} rows of synthetic data")
print(f"Date range: {data.index.min()} to {data.index.max()}")
print(f"Stocks: {sorted(data['Ticker'].unique())}")

# Split data
print("\nSplitting data...")
train_data, test_data = split_data(data)

# Initialize
print("\nInitializing environment and agent...")
train_env = TradingEnvironment(train_data, initial_balance=10000, index_ticker='^GSPC')
test_env = TradingEnvironment(test_data, initial_balance=10000, index_ticker='^GSPC')

agent = QLearningAgent(
    state_size=54,
    action_size=5,
    learning_rate=0.01,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01
)

# Train for a few episodes (reduced for quick testing)
print("\nTraining agent (10 episodes for quick test)...")
num_episodes = 10

for episode in range(num_episodes):
    state = train_env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = agent.act(state, training=True)
        next_state, reward, done, info = train_env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
    
    agent.end_episode()
    conv_value = agent.track_convergence()
    
    print(f"Episode {episode + 1:3d} | Reward: {episode_reward:8.2f} | "
          f"Portfolio: ${info['portfolio_value']:10.2f} | "
          f"Convergence: {conv_value:.6f if conv_value else 'N/A'}")

# Test
print("\nTesting agent on test data...")
state = test_env.reset()
done = False
test_portfolio_values = [test_env.initial_balance]

while not done:
    action = agent.act(state, training=False)
    next_state, reward, done, info = test_env.step(action)
    state = next_state
    test_portfolio_values.append(info['portfolio_value'])

final_value = test_portfolio_values[-1]
total_return = (final_value - test_env.initial_balance) / test_env.initial_balance * 100

print(f"\nTest Results:")
print(f"  Initial: ${test_env.initial_balance:,.2f}")
print(f"  Final: ${final_value:,.2f}")
print(f"  Return: {total_return:.2f}%")

print("\n[OK] Training and testing completed successfully!")
print("You can now use train.py with your real data file.")

