## Test Technical Indicators

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))
from utils import *

# Create synthetic price data for testing
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
dates = dates[dates.weekday < 5]  # Only weekdays (trading days)

# Generate realistic price data with trend and volatility
n_days = len(dates)
base_price = 100
trend = np.linspace(0, 50, n_days)  # Upward trend
noise = np.random.randn(n_days).cumsum() * 2
prices = base_price + trend + noise
prices = pd.Series(prices, index=dates)

# Generate index data (S&P 500-like)
index_prices = base_price + trend * 0.8 + noise * 0.7
index_prices = pd.Series(index_prices, index=dates)

print("=" * 60)
print("TESTING TECHNICAL INDICATORS")
print("=" * 60)

# Test 1: Calculate Returns
print("\n1. Testing calculate_returns()...")
returns = calculate_returns(prices)
print(f"   ✓ Returns calculated: {len(returns)} values")
print(f"   First 5 returns: {returns.head().values}")
print(f"   Mean return: {returns.mean():.4f}")

# Test 2: Calculate SMA
print("\n2. Testing calculate_sma()...")
sma_50 = calculate_sma(prices, window=50)
print(f"   ✓ SMA_50 calculated: {len(sma_50)} values")
print(f"   First 50 values are NaN: {sma_50.iloc[:50].isna().all()}")
print(f"   First non-NaN value (index 50): {sma_50.iloc[50]:.2f}")
print(f"   Last SMA value: {sma_50.iloc[-1]:.2f}")
print(f"   Last price: {prices.iloc[-1]:.2f}")

# Test 3: Calculate Volatility
print("\n3. Testing calculate_volatility()...")
vol_30 = calculate_volatility(returns, window=30)
print(f"   ✓ 30-day volatility calculated: {len(vol_30)} values")
print(f"   First 30 values are NaN: {vol_30.iloc[:30].isna().all()}")
print(f"   Mean 30d volatility: {vol_30.mean():.4f}")
print(f"   Last 30d volatility: {vol_30.iloc[-1]:.4f}")

# Test 4: Calculate Relative Volatility
print("\n4. Testing calculate_relative_volatility()...")
rel_vol = calculate_relative_volatility(returns, short_window=30, long_window=365)
print(f"   ✓ Relative volatility calculated")
print(f"   Shape: {rel_vol.shape}")
print(f"   Columns: {list(rel_vol.columns)}")
print(f"   First 365 values have NaN for vol_long: {rel_vol['vol_long'].iloc[:365].isna().all()}")
print(f"   Last relative_vol value: {rel_vol['relative_vol'].iloc[-1]:.4f}")
print(f"   Mean relative_vol (where not NaN): {rel_vol['relative_vol'].mean():.4f}")

# Test 5: Calculate Trend Indicator
print("\n5. Testing calculate_trend_indicator()...")
trend_ind = calculate_trend_indicator(prices, sma_window=50)
print(f"   ✓ Trend indicator calculated: {len(trend_ind)} values")
print(f"   First 50 values are NaN: {trend_ind.iloc[:50].isna().all()}")
print(f"   Last trend value: {trend_ind.iloc[-1]:.2f}%")
print(f"   Mean trend (where not NaN): {trend_ind.mean():.2f}%")

# Test 6: Calculate Portfolio Return
print("\n6. Testing calculate_portfolio_return()...")
# Simulate portfolio values (starting at 10000)
portfolio_values = pd.Series(10000 * (1 + returns).cumprod(), index=dates)
portfolio_ret = calculate_portfolio_return(portfolio_values, window=30)
print(f"   ✓ Portfolio return calculated: {len(portfolio_ret)} values")
print(f"   First 30 values are NaN: {portfolio_ret.iloc[:30].isna().all()}")
print(f"   Last 30d portfolio return: {portfolio_ret.iloc[-1]:.4f} ({portfolio_ret.iloc[-1]*100:.2f}%)")

# Test 7: Calculate Index Return
print("\n7. Testing calculate_index_return()...")
index_ret = calculate_index_return(index_prices, window=30)
print(f"   ✓ Index return calculated: {len(index_ret)} values")
print(f"   Last 30d index return: {index_ret.iloc[-1]:.4f} ({index_ret.iloc[-1]*100:.2f}%)")

# Test 8: Calculate Performance Difference
print("\n8. Testing calculate_performance_difference()...")
perf_diff = calculate_performance_difference(portfolio_values, index_prices, window=30)
print(f"   ✓ Performance difference calculated: {len(perf_diff)} values")
print(f"   Last performance difference: {perf_diff.iloc[-1]:.2f} percentage points")
print(f"   Mean performance diff (where not NaN): {perf_diff.mean():.2f} pp")

# Visualizations
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Technical Indicators Test Results', fontsize=16)

# Plot 1: Prices and SMA
axes[0, 0].plot(prices.index, prices.values, label='Price', alpha=0.7)
axes[0, 0].plot(sma_50.index, sma_50.values, label='SMA_50', linewidth=2)
axes[0, 0].set_title('Price and SMA_50')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Trend Indicator
axes[0, 1].plot(trend_ind.index, trend_ind.values, color='green', linewidth=1.5)
axes[0, 1].axhline(y=2, color='r', linestyle='--', label='Bullish threshold (+2%)')
axes[0, 1].axhline(y=-2, color='r', linestyle='--', label='Bearish threshold (-2%)')
axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[0, 1].set_title('Trend Indicator: (Price - SMA_50) / Price')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Percentage (%)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Volatility
axes[1, 0].plot(vol_30.index, vol_30.values, label='30d Volatility', color='orange')
axes[1, 0].set_title('30-Day Rolling Volatility (Annualized)')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Volatility')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Relative Volatility
axes[1, 1].plot(rel_vol.index, rel_vol['relative_vol'].values, color='purple', linewidth=1.5)
axes[1, 1].axhline(y=1, color='r', linestyle='--', label='Threshold (σ_30d = σ_365d)')
axes[1, 1].set_title('Relative Volatility: σ_30d / σ_365d')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Ratio')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Returns
axes[2, 0].plot(returns.index, returns.values, alpha=0.6, color='blue')
axes[2, 0].set_title('Daily Returns')
axes[2, 0].set_xlabel('Date')
axes[2, 0].set_ylabel('Return')
axes[2, 0].grid(True, alpha=0.3)

# Plot 6: Performance Difference
axes[2, 1].plot(perf_diff.index, perf_diff.values, color='red', linewidth=1.5)
axes[2, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[2, 1].axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Overperforming threshold')
axes[2, 1].axhline(y=-0.5, color='r', linestyle='--', alpha=0.5, label='Underperforming threshold')
axes[2, 1].set_title('Portfolio Performance vs Index (30d)')
axes[2, 1].set_xlabel('Date')
axes[2, 1].set_ylabel('Difference (percentage points)')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ All tests completed successfully!")
print("\nNext steps:")
print("  - Verify that NaN values appear in the first N periods (where N = window size)")
print("  - Check that trend indicator values are in reasonable ranges")
print("  - Verify relative volatility ratio makes sense (typically 0.5 to 2.0)")