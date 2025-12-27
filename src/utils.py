"""
Utility functions for the trading project.
This module contains helper functions for data processing, visualization, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filepath):
    """
    Load trading data from file.
    
    Args:
        filepath: Path to data file
        
    Returns:
        DataFrame with market data
    """
    df = pd.read_csv(filepath)

    # If a date column exists, parse and sort for time-series consistency
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.sort_values(col)
            break

    return df.reset_index(drop=True)


def normalize_data(data):
    """
    Normalize data for better training.
    
    Args:
        data: Raw data to normalize
        
    Returns:
        Normalized data
    """
    df = data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Z-score normalization; avoid division by zero
    for col in numeric_cols:
        std = df[col].std()
        if std == 0 or pd.isna(std):
            df[col] = 0.0
        else:
            df[col] = (df[col] - df[col].mean()) / std

    return df


def calculate_returns(prices):
    """
    Calculate returns from price data.
    
    Args:
        prices: Price series
        
    Returns:
        Returns series
    """
    return prices.pct_change().dropna()


def plot_portfolio_performance(portfolio_values, title="Portfolio Performance"):
    """
    Plot portfolio value over time.
    
    Args:
        portfolio_values: List or array of portfolio values
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.show()


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Risk-free rate
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_max_drawdown(portfolio_values):
    """
    Calculate maximum drawdown.
    
    Args:
        portfolio_values: Portfolio value series
        
    Returns:
        Maximum drawdown percentage
    """
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    return drawdown.min() * 100

def calculate_sma(prices, window=50):
    """
    Calculate Simple Moving Average (SMA).
    
    SMA is the average of the last N values. When a new value enters,
    the oldest one gets out.
    
    Args:
        prices: Price series (pandas Series or DataFrame)
        window: Number of periods for moving average (default: 50)
        
    Returns:
        Series with SMA values (NaN for first window-1 values)
    """
    return prices.rolling(window=window).mean()


def calculate_volatility(returns, window=30):
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Args:
        returns: Returns series (pandas Series or DataFrame)
        window: Number of periods for rolling window (default: 30)
        
    Returns:
        Series with rolling volatility values (NaN for first window-1 values)
    """
    # Annualized volatility: std * sqrt(252 trading days)
    return returns.rolling(window=window).std() * np.sqrt(252)


def calculate_relative_volatility(returns, short_window=30, long_window=365):
    """
    Calculate relative volatility: σ_short vs σ_long.
    
    This compares short-term volatility (30 days) against long-term volatility (365 days)
    to determine if current market conditions are more volatile than usual.
    
    Args:
        returns: Returns series (pandas Series or DataFrame)
        short_window: Short-term window (default: 30 days)
        long_window: Long-term window (default: 365 days)
        
    Returns:
        DataFrame with columns:
        - 'vol_short': Short-term volatility (30d)
        - 'vol_long': Long-term volatility (365d)
        - 'relative_vol': Ratio of short/long volatility
    """
    vol_short = calculate_volatility(returns, window=short_window)
    vol_long = calculate_volatility(returns, window=long_window)
    
    result = pd.DataFrame({
        'vol_short': vol_short,
        'vol_long': vol_long,
        'relative_vol': vol_short / vol_long
    })
    
    return result


def calculate_trend_indicator(prices, sma_window=50):
    """
    Calculate trend indicator: (Current Price - SMA) / Current Price.
    
    This measures how far the current price is from the moving average,
    expressed as a percentage.
    
    Args:
        prices: Price series (pandas Series or DataFrame)
        sma_window: Window for SMA calculation (default: 50)
        
    Returns:
        Series with trend indicator values (percentage difference)
    """
    sma = calculate_sma(prices, window=sma_window)
    trend = (prices - sma) / prices * 100  # Convert to percentage
    return trend


def calculate_portfolio_return(portfolio_values, window=30):
    """
    Calculate portfolio return over a specified window.
    
    Args:
        portfolio_values: Series of portfolio values over time
        window: Number of periods to look back (default: 30 days)
        
    Returns:
        Series with rolling returns (NaN for first window values)
    """
    # Calculate percentage change over the window
    return portfolio_values.pct_change(periods=window)


def calculate_index_return(index_prices, window=30):
    """
    Calculate index return over a specified window.
    
    Args:
        index_prices: Series of index prices (e.g., S&P 500)
        window: Number of periods to look back (default: 30 days)
        
    Returns:
        Series with rolling returns (NaN for first window values)
    """
    return index_prices.pct_change(periods=window)


def calculate_performance_difference(portfolio_values, index_prices, window=30):
    """
    Calculate portfolio performance vs index: R_portfolio - R_index.
    
    This compares portfolio returns against index returns over a rolling window.
    
    Args:
        portfolio_values: Series of portfolio values over time
        index_prices: Series of index prices (e.g., S&P 500)
        window: Number of periods to look back (default: 30 days)
        
    Returns:
        Series with performance difference (portfolio return - index return)
    """
    portfolio_returns = calculate_portfolio_return(portfolio_values, window=window)
    index_returns = calculate_index_return(index_prices, window=window)
    
    # Convert to percentage points
    performance_diff = (portfolio_returns - index_returns) * 100
    return performance_diff


# State Discretization Functions

def discretize_cash_percentage(cash_percentage):
    """
    Discretize cash percentage to state value (0, 1, or 2).
    
    According to QLearningStructure.md:
    - [0] Invested: < 10% cash in portfolio
    - [1] Balanced: 10 - 50% cash in portfolio (inclusive)
    - [2] Capital: > 50% cash in portfolio
    
    Args:
        cash_percentage: Cash percentage (0-100)
        
    Returns:
        Discrete state value: 0, 1, or 2
    """
    if cash_percentage < 10:
        return 0  # Invested
    elif cash_percentage <= 50:
        return 1  # Balanced
    else:
        return 2  # Capital


def discretize_trend(trend_percentage):
    """
    Discretize trend indicator to state value (0, 1, or 2).
    
    According to QLearningStructure.md:
    - [0] Bearish: < -2%
    - [1] Neutral: in the range ±2% (inclusive)
    - [2] Bullish: > +2%
    
    Args:
        trend_percentage: Trend percentage from calculate_trend_indicator()
                          (Current Price - SMA_50) / Current Price * 100
        
    Returns:
        Discrete state value: 0, 1, or 2
    """
    if trend_percentage < -2:
        return 0  # Bearish
    elif trend_percentage <= 2:
        return 1  # Neutral
    else:
        return 2  # Bullish


def discretize_volatility(relative_vol):
    """
    Discretize relative volatility to state value (0 or 1).
    
    According to QLearningStructure.md:
    - [0] Normal: σ_30d ≤ σ_365d
    - [1] High Risk: σ_30d > σ_365d
    
    Args:
        relative_vol: Relative volatility ratio (σ_30d / σ_365d)
                      from calculate_relative_volatility()
        
    Returns:
        Discrete state value: 0 or 1
    """
    if relative_vol <= 1.0:
        return 0  # Normal
    else:
        return 1  # High Risk


def discretize_performance(performance_diff):
    """
    Discretize portfolio performance to state value (0, 1, or 2).
    
    According to QLearningStructure.md:
    - [0] Underperforming: R_portfolio < R_index (difference < -0.5)
    - [1] Neutral: R_portfolio ≈ R_index (difference in range ±0.5)
    - [2] Overperforming: R_portfolio > R_index (difference > +0.5)
    
    Note: The spec says "R_portfolio < R_index (±0.5)" for neutral,
    which we interpret as |difference| ≤ 0.5
    
    Args:
        performance_diff: Performance difference in percentage points
                         from calculate_performance_difference()
        
    Returns:
        Discrete state value: 0, 1, or 2
    """
    if performance_diff < -0.5:
        return 0  # Underperforming
    elif performance_diff <= 0.5:
        return 1  # Neutral
    else:
        return 2  # Overperforming


def state_tuple_to_index(cash_state, trend_state, vol_state, perf_state):
    """
    Convert discrete state tuple to a single integer index for Q-table lookup.
    
    State dimensions:
    - Cash: 3 values (0, 1, 2)
    - Trend: 3 values (0, 1, 2)
    - Volatility: 2 values (0, 1)
    - Performance: 3 values (0, 1, 2)
    
    Total states = 3 × 3 × 2 × 3 = 54 states
    
    Args:
        cash_state: Cash state (0, 1, or 2)
        trend_state: Trend state (0, 1, or 2)
        vol_state: Volatility state (0 or 1)
        perf_state: Performance state (0, 1, or 2)
        
    Returns:
        Integer index from 0 to 53 for Q-table lookup
    """
    # Validate inputs
    if not (0 <= cash_state <= 2):
        raise ValueError(f"cash_state must be 0, 1, or 2, got {cash_state}")
    if not (0 <= trend_state <= 2):
        raise ValueError(f"trend_state must be 0, 1, or 2, got {trend_state}")
    if not (0 <= vol_state <= 1):
        raise ValueError(f"vol_state must be 0 or 1, got {vol_state}")
    if not (0 <= perf_state <= 2):
        raise ValueError(f"perf_state must be 0, 1, or 2, got {perf_state}")
    
    # Convert to index: cash * 18 + trend * 6 + vol * 3 + perf
    # Where: 18 = 3*2*3, 6 = 2*3, 3 = 3
    index = cash_state * 18 + trend_state * 6 + vol_state * 3 + perf_state
    
    return int(index)


def index_to_state_tuple(index):
    """
    Convert Q-table index back to discrete state tuple.
    
    Inverse function of state_tuple_to_index().
    
    Args:
        index: Integer index from 0 to 53
        
    Returns:
        Tuple (cash_state, trend_state, vol_state, perf_state)
    """
    if not (0 <= index < 54):
        raise ValueError(f"index must be between 0 and 53, got {index}")
    
    perf_state = index % 3
    vol_state = (index // 3) % 2
    trend_state = (index // 6) % 3
    cash_state = index // 18
    
    return (cash_state, trend_state, vol_state, perf_state)


# Portfolio Management Functions


def calculate_stock_returns(data, current_step, lookback_window=30):
    """
    Calculate returns for each stock over a lookback window.
    
    Args:
        data: DataFrame with Date index and Ticker column
        current_step: Current time step index
        lookback_window: Number of days to look back (default: 30)
        
    Returns:
        Series with ticker as index and returns as values
    """
    if current_step < lookback_window:
        return pd.Series(dtype=float)  # Not enough data
    
    # Get data for the lookback period
    start_idx = max(0, current_step - lookback_window)
    end_idx = current_step + 1
    
    period_data = data.iloc[start_idx:end_idx]
    
    # Group by ticker and calculate returns
    returns_dict = {}
    for ticker in period_data['Ticker'].unique():
        ticker_data = period_data[period_data['Ticker'] == ticker]
        if len(ticker_data) < 2:
            continue
        
        # Get first and last close price
        prices = ticker_data['Close'].values
        if len(prices) >= 2:
            return_pct = (prices[-1] - prices[0]) / prices[0]
            returns_dict[ticker] = return_pct
    
    return pd.Series(returns_dict)


def get_top_performing_stocks(data, current_step, top_n=5, lookback_window=30):
    """
    Get the top N performing stocks over a lookback window.
    
    Buy Policy: Buy the Top 5 performing stocks split equally.
    
    Args:
        data: DataFrame with Date index and Ticker column
        current_step: Current time step index
        top_n: Number of top stocks to return (default: 5)
        lookback_window: Number of days to look back (default: 30)
        
    Returns:
        List of ticker symbols (top performers)
    """
    returns = calculate_stock_returns(data, current_step, lookback_window)
    
    if len(returns) == 0:
        return []
    
    # Sort by returns (descending) and get top N
    top_stocks = returns.nlargest(top_n).index.tolist()
    
    return top_stocks


def get_worst_performing_stock(data, current_step, holdings, lookback_window=30):
    """
    Get the worst performing stock from current holdings.
    
    Sell Policy: Sell the worst performing stock until we reach the cap.
    
    Args:
        data: DataFrame with Date index and Ticker column
        current_step: Current time step index
        holdings: Dictionary of current holdings {ticker: shares}
        lookback_window: Number of days to look back (default: 30)
        
    Returns:
        Ticker symbol of worst performing stock, or None if no holdings
    """
    if len(holdings) == 0:
        return None
    
    returns = calculate_stock_returns(data, current_step, lookback_window)
    
    # Filter to only stocks we own
    owned_returns = returns[returns.index.isin(holdings.keys())]
    
    if len(owned_returns) == 0:
        # If we can't calculate returns, return first stock
        return list(holdings.keys())[0]
    
    # Return worst performer (lowest return)
    worst_stock = owned_returns.idxmin()
    
    return worst_stock


def calculate_portfolio_value(holdings, cash, data, current_step):
    """
    Calculate total portfolio value (cash + stock holdings).
    
    Args:
        holdings: Dictionary {ticker: shares}
        cash: Current cash balance
        data: DataFrame with Date index and Ticker column
        current_step: Current time step index
        
    Returns:
        Total portfolio value
    """
    stock_value = 0.0
    
    # Get the date at current_step
    current_date = data.index[current_step]
    
    # Get all rows for this date
    date_data = data.loc[current_date]
    
    # Handle both Series and DataFrame cases
    if isinstance(date_data, pd.Series):
        # Single row for this date
        ticker = date_data.get('Ticker')
        if ticker in holdings:
            price = date_data.get('Close', 0)
            stock_value += holdings[ticker] * price
    else:
        # Multiple rows (one per ticker)
        for ticker, shares in holdings.items():
            ticker_data = date_data[date_data['Ticker'] == ticker]
            if len(ticker_data) > 0:
                price = ticker_data['Close'].iloc[0]
                stock_value += shares * price
    
    return cash + stock_value


def calculate_cash_percentage(cash, portfolio_value):
    """
    Calculate cash as percentage of total portfolio value.
    
    Args:
        cash: Current cash balance
        portfolio_value: Total portfolio value
        
    Returns:
        Cash percentage (0-100)
    """
    if portfolio_value == 0:
        return 100.0  # All cash if portfolio is empty
    
    return (cash / portfolio_value) * 100.0