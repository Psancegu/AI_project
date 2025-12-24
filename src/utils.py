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