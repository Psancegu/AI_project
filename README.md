# Trading Q-Learning Project

This project implements a Q-learning algorithm for trading strategies.

## Project Structure

```
AI_project/
├── data/              # Data files
├── notebooks/         # Jupyter notebooks
│   └── trading_qlearning.ipynb
├── src/               # Source code
│   ├── environment.py  # Trading environment
│   ├── qlearning.py    # Q-learning agent
│   └── utils.py        # Utility functions
└── README.md
```

## Setup

1. Install required dependencies:
```bash
pip install numpy pandas matplotlib jupyter
```

2. Place your trading data in the `data/` directory.

3. Open `notebooks/trading_qlearning.ipynb` to start working.

## Usage

See the main notebook for detailed usage instructions.

## Missing components:


1. State discretization
- Current: get_state() returns a continuous numpy array.
- Needed: A function to map continuous features to a discrete state tuple (cash, trend, volatility, performance) where each value is 0-2 (or 0-1 for volatility).

- Missing calculations:
    - Cash percentage: cash / total_portfolio_value
    - Trend: (current_price - SMA_50) / current_price → discretize to 0-2
    - Relative volatility: σ_30d vs σ_365d → discretize to 0-1
    - Portfolio performance: R_portfolio - R_index (30-day) → discretize to 0-2

2. Action space
- Current: Only 3 actions (hold, buy, sell) mapped to position values.
- Needed: 5 actions:
[0] Hold
[1] Buy Conservative (25% of cash)
[2] Buy Aggressive (100% of cash)
[3] Sell Conservative (25% of holdings)
[4] Sell Aggressive (100% of holdings)

3. Reward function
- Current: Simple price_change * position.
- Needed: The 3-part reward from the spec:
    - Return component with 1.5× multiplier for losses
    - Commission penalty (-0.1 for non-hold actions)
    - Active portfolio reward (0-0.04 based on cash percentage)

4. Portfolio management
- Current: Single position tracking (self.position = 0/1/-1).
- Needed:
    - Track multiple stocks (holdings dictionary)
    - Buy policy: Top 5 performing stocks split equally
    - Sell policy: Worst performing stock until cap reached
    - Track cash vs invested capital separately

5. State-to-index mapping
- Current: Q-table uses continuous state indices.
- Needed: A function to convert the discrete state tuple (cash, trend, vol, perf) to a single integer index (0-53) for Q-table lookup.

6. Technical indicators
- Missing calculations:
- SMA_50 (Simple Moving Average, 50 days)
- σ_30d and σ_365d (volatility calculations)
- Portfolio return vs index return (30-day comparison)

7. Q-table initialization
- Current: np.zeros((state_size, action_size)) with continuous state_size.
- Needed: Q-table with shape (54, 5) for 54 discrete states × 5 actions.

8. Convergence tracking
- Missing: Function to compute sum of absolute differences between consecutive Q-tables for convergence detection.

9. Data split
- Missing: Logic to split data into training (2000-2021) and test (2022+) periods.

The current implementation is a basic RL environment, but it doesn't match the discrete state space, multi-stock portfolio, and reward function specified in QLearningStructure.md.