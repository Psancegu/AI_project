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

