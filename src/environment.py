"""
Environment module for trading simulation.
This module defines the trading environment for Q-learning.
"""

import numpy as np


class TradingEnvironment:
    """
    Trading environment for reinforcement learning.
    """
    
    def __init__(self, data, initial_balance=10000):
        """
        Initialize the trading environment.
        
        Args:
            data: Market data (prices, volumes, etc.)
            initial_balance: Starting capital
        """
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state observation
        """
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        
        return self.get_state()
    
    def get_state(self):
        """
        Get current state observation.
        
        Returns:
            State vector as a 1D numpy array containing:
            [balance, position, portfolio_value, <market features>]
        """
        # Extract current market features (numeric columns only)
        row = self.data.iloc[self.current_step]
        market_features = row.select_dtypes(include=[np.number]).to_numpy()

        # Combine account status with market features
        state = np.concatenate(
            [
                np.array([self.balance, self.position, self.portfolio_value], dtype=float),
                market_features.astype(float),
            ]
        )

        return state
    
    def step(self, action):
        """

        funció a revisar: el reward ignora la comisió de la operació, la comisió es del 0.1% de la operació.
        no hi ha noció de quants stocks es poden comprar o vendre, es pot comprar o vendre tantes com es vulgui.
        a més, s'ignora la posicio tamany/ capital, multiplica el canvi de preu per la posicio per obtenir el reward.
        Execute an action in the environment.
        
        Args:
            action: Action to take (buy, sell, hold)
            
        Returns:
            next_state, reward, done, info
        """
        # Map actions: 0=hold, 1=buy, 2=sell
        # despés afegir mes accions, de moment les tres basiques per provar
        action_map = {0: 0, 1: 1, 2: -1}
        if action not in action_map:
            raise ValueError(f"Invalid action {action}. Expected one of {list(action_map.keys())}.")

        # Current price (before moving to next step)
        current_price = self._get_price(self.current_step)

        # Advance time
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Next price for reward calculation
        next_price = self._get_price(self.current_step)
        price_change = next_price - current_price

        # Reward based on held position during this interval
        reward = price_change * self.position

        # Update position according to action (applied for next interval)
        self.position = action_map[action]

        # Update account metrics (unrealized PnL captured in balance)
        self.balance += reward
        self.portfolio_value = self.balance

        next_state = self.get_state()
        info = {"price": next_price, "balance": self.balance, "position": self.position}

        return next_state, reward, done, info
    
    def render(self):
        """
        Render the current state of the environment.
        """
        # TODO: Implement visualization
        pass
    

    def _get_price(self, idx):
        """
        Helper to extract the price at a given index. Prefers common 'close'
        naming but falls back to the first numeric column.
        """
        row = self.data.iloc[idx]

        # Case-insensitive lookup for a closing price
        lower_map = {col.lower(): col for col in row.index}
        for key in ("close", "adj close", "adj_close"):
            if key in lower_map:
                return float(row[lower_map[key]])

        # Fallback: first numeric column (e.g., Open)
        numeric_cols = row.select_dtypes(include=[np.number])
        if len(numeric_cols) == 0:
            raise ValueError("Data must contain at least one numeric price column.")

        return float(numeric_cols.iloc[0])
