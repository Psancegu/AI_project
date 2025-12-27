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
    calculate_sma,
    calculate_trend_indicator,
    calculate_returns,
    calculate_relative_volatility,
    calculate_performance_difference,
    discretize_cash_percentage,
    discretize_trend,
    discretize_volatility,
    discretize_performance,
    state_tuple_to_index,
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
        
        # Pre-compute index data for efficiency
        self._prepare_index_data()
        
        return self.get_state()
    
    def _prepare_index_data(self):
        """
        Pre-compute index prices and technical indicators for efficiency.
        This is called once during reset.
        """
        # Extract index prices
        index_data = self.data[self.data['Ticker'] == self.index_ticker].copy()
        
        if len(index_data) == 0:
            # If index not found, create a dummy series with constant price
            unique_dates = self.data.index.unique()
            self.index_prices = pd.Series([100.0] * len(unique_dates), index=unique_dates)
        else:
            # Create a dictionary mapping dates to Close prices
            index_dict = {}
            for date in index_data.index.unique():
                date_rows = index_data.loc[date]
                if isinstance(date_rows, pd.Series):
                    index_dict[date] = date_rows.get('Close', 100.0)
                else:
                    # Multiple rows for same date, take first
                    index_dict[date] = date_rows['Close'].iloc[0] if len(date_rows) > 0 else 100.0
            
            # Create Series from dictionary
            self.index_prices = pd.Series(index_dict)
        
        # Calculate index returns for volatility calculation
        self.index_returns = calculate_returns(self.index_prices)
    
    def _get_index_price_at_step(self, step):
        """
        Get index price at a specific step.
        
        Args:
            step: Step index
            
        Returns:
            Index price, or 100.0 if not available
        """
        if step >= len(self.data.index.unique()):
            return 100.0
        
        date = self.data.index[step]
        if date in self.index_prices.index:
            return float(self.index_prices.loc[date])
        return 100.0
    
    def get_state(self):
        """
        Get current discrete state observation.
        
        Calculates the 4-dimensional state:
        - Cash percentage (0-2)
        - Trend indicator (0-2)
        - Relative volatility (0-1)
        - Portfolio performance (0-2)
        
        Returns:
            Discrete state index (0-53) for Q-table lookup
        """
        # Default state values (neutral/middle values)
        cash_state = 1  # Balanced
        trend_state = 1  # Neutral
        vol_state = 0  # Normal
        perf_state = 1  # Neutral
        
        # 1. Calculate cash percentage
        try:
            portfolio_value = calculate_portfolio_value(
                self.holdings, self.cash, self.data, self.current_step
            )
            cash_pct = calculate_cash_percentage(self.cash, portfolio_value)
            cash_state = discretize_cash_percentage(cash_pct)
        except Exception:
            cash_state = 1  # Default to balanced
        
        # 2. Calculate trend indicator (need at least 50 days of data)
        try:
            if self.current_step >= 50:
                # Get index prices up to current step
                date = self.data.index[self.current_step]
                dates_up_to_now = self.data.index[:self.current_step + 1].unique()
                
                # Get index prices for these dates
                index_prices_series = pd.Series(
                    [self._get_index_price_at_step(i) 
                     for i in range(min(51, self.current_step + 1))],
                    index=dates_up_to_now[:min(51, self.current_step + 1)]
                )
                
                if len(index_prices_series) >= 50:
                    trend_series = calculate_trend_indicator(index_prices_series, sma_window=50)
                    if len(trend_series) > 0 and not pd.isna(trend_series.iloc[-1]):
                        trend_pct = float(trend_series.iloc[-1])
                        trend_state = discretize_trend(trend_pct)
        except Exception:
            trend_state = 1  # Default to neutral
        
        # 3. Calculate relative volatility (need at least 365 days of data)
        try:
            if self.current_step >= 365:
                # Get index returns up to current step
                date = self.data.index[self.current_step]
                dates_up_to_now = self.data.index[:self.current_step + 1].unique()
                
                # Get index prices for enough history
                index_prices_series = pd.Series(
                    [self._get_index_price_at_step(i) 
                     for i in range(min(366, self.current_step + 1))],
                    index=dates_up_to_now[:min(366, self.current_step + 1)]
                )
                
                if len(index_prices_series) >= 365:
                    index_returns_series = calculate_returns(index_prices_series)
                    rel_vol_df = calculate_relative_volatility(
                        index_returns_series, short_window=30, long_window=365
                    )
                    if len(rel_vol_df) > 0 and not pd.isna(rel_vol_df['relative_vol'].iloc[-1]):
                        rel_vol = float(rel_vol_df['relative_vol'].iloc[-1])
                        vol_state = discretize_volatility(rel_vol)
        except Exception:
            vol_state = 0  # Default to normal
        
        # 4. Calculate portfolio performance vs index (need at least 30 days of history)
        try:
            if len(self.portfolio_history) >= 30:
                # Get portfolio values as Series
                portfolio_series = pd.Series(self.portfolio_history)
                
                # Get index prices for the same period
                dates_for_history = self.data.index[:len(self.portfolio_history)].unique()
                index_prices_series = pd.Series(
                    [self._get_index_price_at_step(i) 
                     for i in range(len(self.portfolio_history))],
                    index=dates_for_history[:len(self.portfolio_history)]
                )
                
                if len(portfolio_series) >= 30 and len(index_prices_series) >= 30:
                    perf_diff_series = calculate_performance_difference(
                        portfolio_series, index_prices_series, window=30
                    )
                    if len(perf_diff_series) > 0 and not pd.isna(perf_diff_series.iloc[-1]):
                        perf_diff = float(perf_diff_series.iloc[-1])
                        perf_state = discretize_performance(perf_diff)
        except Exception:
            perf_state = 1  # Default to neutral
        
        # Convert state tuple to index
        state_index = state_tuple_to_index(cash_state, trend_state, vol_state, perf_state)
        
        return state_index
    
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
        
        # Calculate reward using the 3-part reward function from QLearningStructure.md
        # R_t = Return_component - Commission_penalty + Active_portfolio_reward
        
        # 1. Return component (with loss aversion: 1.5x multiplier for losses)
        if prev_portfolio_value > 0:
            portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            portfolio_return = 0.0
        
        if portfolio_return >= 0:
            return_component = portfolio_return
        else:
            return_component = 1.5 * portfolio_return  # Loss aversion
        
        # 2. Commission penalty (0.1 for non-hold actions)
        if action != 0:
            commission_penalty = 0.1
        else:
            commission_penalty = 0.0
        
        # 3. Active portfolio reward (based on cash percentage)
        cash_pct = calculate_cash_percentage(self.cash, current_portfolio_value)
        cash_state = discretize_cash_percentage(cash_pct)
        
        if cash_state == 0:  # < 10% cash (Invested)
            active_reward = 0.04
        elif cash_state == 1:  # 10-50% cash (Balanced)
            active_reward = 0.02
        else:  # > 50% cash (Capital)
            active_reward = 0.0
        
        # Total reward
        reward = return_component - commission_penalty + active_reward
        
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
