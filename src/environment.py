"""
Environment module for trading simulation.
This module defines the trading environment for Q-learning.
"""

import numpy as np
import pandas as pd
from utils import (
    get_top_performing_stocks,
    get_worst_performing_stock,
    calculate_portfolio_value,
    calculate_cash_percentage,
)


class TradingEnvironment:
    """
    Trading environment for reinforcement learning.
    """
    
    def __init__(self, data, initial_balance=10000, index_ticker='^GSPC'):
        """
        Initialize the trading environment.
        
        Args:
            data: Market data DataFrame with Date index and Ticker column
            initial_balance: Starting capital
            index_ticker: Ticker symbol for the index (default: '^GSPC' for S&P 500)
        """
        self.data = data
        self.initial_balance = initial_balance
        self.index_ticker = index_ticker
        
        # Verify data structure
        if 'Ticker' not in data.columns:
            raise ValueError("Data must have a 'Ticker' column")
        
        self.reset()
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial discrete state index (0-53)
        """
        self.cash = self.initial_balance
        self.holdings = {}  # Dictionary: {ticker: shares}
        self.current_step = 0
        self.portfolio_history = [self.initial_balance]  # Track portfolio value over time
        
        return self.get_state()
    
    def get_state(self):
        """
        Get current discrete state observation.
        
        Returns:
            Discrete state index (0-53) for Q-table lookup
        """
        # This will be implemented after we add technical indicators
        # For now, return a placeholder
        # TODO: Calculate discrete state from technical indicators
        return 0
    
    def _get_current_prices(self):
        """
        Get current prices for all stocks at current_step.
        
        Returns:
            Dictionary {ticker: price}
        """
        # Get the date at current_step
        current_date = self.data.index[self.current_step]
        
        # Filter data for this specific date - always get as DataFrame
        date_data = self.data.loc[current_date]
        
        prices = {}
        
        # Convert to DataFrame if it's a Series
        if isinstance(date_data, pd.Series):
            # Single row - convert to DataFrame for consistent handling
            date_data = date_data.to_frame().T
        
        # Now date_data is always a DataFrame
        if 'Ticker' in date_data.columns and 'Close' in date_data.columns:
            for _, row in date_data.iterrows():
                ticker = row['Ticker']
                close_price = row['Close']
                prices[ticker] = close_price
        
        return prices
    
    def _get_stock_price(self, ticker):
        """
        Get current price for a specific stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Current price, or 0 if not found
        """
        prices = self._get_current_prices()
        return prices.get(ticker, 0)
    
    def _buy_stocks(self, amount_to_invest, conservative=True):
        """
        Buy stocks according to buy policy.
        
        Buy Policy: Buy the Top 5 performing stocks split equally.
        
        Args:
            amount_to_invest: Amount of cash to invest
            conservative: If True, buy 25% of cash; if False, buy 100% of cash
            
        Returns:
            Amount actually spent (after commissions)
        """
        if amount_to_invest <= 0:
            return 0
        
        # Get top 5 performing stocks
        top_stocks = get_top_performing_stocks(self.data, self.current_step, top_n=5)
        
        if len(top_stocks) == 0:
            return 0  # No stocks available
        
        # Split investment equally among top stocks
        amount_per_stock = amount_to_invest / len(top_stocks)
        total_spent = 0
        commission_rate = 0.001  # 0.1% commission
        
        for ticker in top_stocks:
            price = self._get_stock_price(ticker)
            if price <= 0:
                continue
            
            # Calculate shares we can buy (accounting for commission)
            # shares * price * (1 + commission) <= amount_per_stock
            shares = amount_per_stock / (price * (1 + commission_rate))
            shares = int(shares)  # Can only buy whole shares
            
            if shares > 0:
                cost = shares * price * (1 + commission_rate)
                if cost <= amount_to_invest - total_spent:
                    # Update holdings
                    self.holdings[ticker] = self.holdings.get(ticker, 0) + shares
                    total_spent += cost
        
        return total_spent
    
    def _sell_stocks(self, percentage_to_sell, conservative=True):
        """
        Sell stocks according to sell policy.
        
        Sell Policy: Sell the worst performing stock until we reach the cap.
        
        Args:
            percentage_to_sell: Percentage of holdings value to sell (0-1)
            conservative: If True, sell 25%; if False, sell 100%
            
        Returns:
            Amount received (after commissions)
        """
        if len(self.holdings) == 0:
            return 0
        
        # Calculate total holdings value
        total_holdings_value = 0
        holdings_value_dict = {}
        
        for ticker, shares in self.holdings.items():
            price = self._get_stock_price(ticker)
            value = shares * price
            holdings_value_dict[ticker] = value
            total_holdings_value += value
        
        if total_holdings_value == 0:
            return 0
        
        # Calculate target amount to sell
        target_value = total_holdings_value * percentage_to_sell
        total_received = 0
        commission_rate = 0.001  # 0.1% commission
        
        # Sell worst performing stocks until we reach target
        while total_received < target_value and len(self.holdings) > 0:
            worst_ticker = get_worst_performing_stock(
                self.data, self.current_step, self.holdings
            )
            
            if worst_ticker is None or worst_ticker not in self.holdings:
                break
            
            shares_to_sell = self.holdings[worst_ticker]
            price = self._get_stock_price(worst_ticker)
            
            if price <= 0:
                # Remove from holdings if price is invalid
                del self.holdings[worst_ticker]
                continue
            
            # Calculate proceeds (after commission)
            proceeds = shares_to_sell * price * (1 - commission_rate)
            
            # Check if selling this would exceed target
            if total_received + proceeds > target_value:
                # Sell partial shares to meet target
                remaining_target = target_value - total_received
                shares_to_sell = int(remaining_target / (price * (1 - commission_rate)))
                proceeds = shares_to_sell * price * (1 - commission_rate)
            
            # Update holdings
            self.holdings[worst_ticker] -= shares_to_sell
            if self.holdings[worst_ticker] <= 0:
                del self.holdings[worst_ticker]
            
            total_received += proceeds
        
        return total_received
    
    def step(self, action):
        """
        Execute an action in the environment.
        
        Actions:
        - [0] Hold: Do Nothing
        - [1] Buy Conservative: Buy 25% of remaining cash in stocks
        - [2] Buy Aggressive: Buy 100% of remaining cash in stocks
        - [3] Sell Conservative: Sell 25% of actives to cash
        - [4] Sell Aggressive: Sell 100% of actives to cash
        
        Args:
            action: Action to take (0-4)
            
        Returns:
            next_state, reward, done, info
        """
        # Store previous portfolio value for reward calculation
        prev_portfolio_value = calculate_portfolio_value(
            self.holdings, self.cash, self.data, self.current_step
        )
        
        # Execute action
        if action == 0:
            # Hold: Do nothing
            amount_spent = 0
            amount_received = 0
        elif action == 1:
            # Buy Conservative: 25% of cash
            amount_to_invest = self.cash * 0.25
            amount_spent = self._buy_stocks(amount_to_invest, conservative=True)
            self.cash -= amount_spent
            amount_received = 0
        elif action == 2:
            # Buy Aggressive: 100% of cash
            amount_to_invest = self.cash
            amount_spent = self._buy_stocks(amount_to_invest, conservative=False)
            self.cash -= amount_spent
            amount_received = 0
        elif action == 3:
            # Sell Conservative: 25% of holdings
            amount_received = self._sell_stocks(0.25, conservative=True)
            self.cash += amount_received
            amount_spent = 0
        elif action == 4:
            # Sell Aggressive: 100% of holdings
            amount_received = self._sell_stocks(1.0, conservative=False)
            self.cash += amount_received
            amount_spent = 0
        else:
            raise ValueError(f"Invalid action {action}. Expected 0-4.")
        
        # Advance time
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Calculate new portfolio value
        current_portfolio_value = calculate_portfolio_value(
            self.holdings, self.cash, self.data, self.current_step
        )
        self.portfolio_history.append(current_portfolio_value)
        
        # Calculate reward (will be implemented with full reward function later)
        # For now, simple return-based reward
        portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
        reward = portfolio_return
        
        # Get next state
        next_state = self.get_state()
        
        info = {
            "portfolio_value": current_portfolio_value,
            "cash": self.cash,
            "holdings": dict(self.holdings),
            "action": action
        }
        
        return next_state, reward, done, info
    
    def render(self):
        """
        Render the current state of the environment.
        """
        # TODO: Implement visualization
        pass
