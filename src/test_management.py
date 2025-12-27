## Test Portfolio Management

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))
from environment import TradingEnvironment
from utils import (
    get_top_performing_stocks,
    get_worst_performing_stock,
    calculate_portfolio_value,
    calculate_cash_percentage,
)

print("=" * 70)
print("TESTING PORTFOLIO MANAGEMENT")
print("=" * 70)

# Create synthetic multi-stock data
print("\n1. Creating synthetic data...")
np.random.seed(42)

# Create date range (100 trading days)
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
dates = dates[dates.weekday < 5]  # Only weekdays

# Create 10 stocks with different price patterns
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT']
n_days = len(dates)

# Generate price data for each stock
data_rows = []
for ticker in tickers:
    # Each stock has different base price and trend
    base_price = np.random.uniform(50, 200)
    trend = np.linspace(0, np.random.uniform(-20, 30), n_days)
    noise = np.random.randn(n_days).cumsum() * np.random.uniform(1, 3)
    prices = base_price + trend + noise
    prices = np.maximum(prices, 1)  # Ensure positive prices
    
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

# Create DataFrame
test_data = pd.DataFrame(data_rows)
test_data = test_data.set_index('Date')

print(f"   ✓ Created data with {len(test_data)} rows")
print(f"   ✓ Date range: {test_data.index.min()} to {test_data.index.max()}")
print(f"   ✓ Stocks: {sorted(test_data['Ticker'].unique())}")
print(f"   ✓ Columns: {list(test_data.columns)}")

# Test 2: Initialize Environment
print("\n2. Testing environment initialization...")
try:
    env = TradingEnvironment(test_data, initial_balance=10000)
    print(f"   ✓ Environment initialized")
    print(f"   ✓ Initial cash: ${env.cash:.2f}")
    print(f"   ✓ Initial holdings: {env.holdings}")
    print(f"   ✓ Current step: {env.current_step}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    raise

# Test 3: Test get_top_performing_stocks
print("\n3. Testing get_top_performing_stocks()...")
try:
    # Need enough history (30+ days)
    env.current_step = 50
    top_stocks = get_top_performing_stocks(test_data, env.current_step, top_n=5)
    print(f"   ✓ Top 5 performing stocks: {top_stocks}")
    print(f"   ✓ Number of stocks returned: {len(top_stocks)}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Test portfolio value calculation
print("\n4. Testing calculate_portfolio_value()...")
try:
    # Start with just cash
    portfolio_value = calculate_portfolio_value(env.holdings, env.cash, test_data, env.current_step)
    print(f"   ✓ Portfolio value (cash only): ${portfolio_value:.2f}")
    print(f"   ✓ Cash: ${env.cash:.2f}")
    print(f"   ✓ Holdings value: ${portfolio_value - env.cash:.2f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 5: Test cash percentage
print("\n5. Testing calculate_cash_percentage()...")
try:
    cash_pct = calculate_cash_percentage(env.cash, portfolio_value)
    print(f"   ✓ Cash percentage: {cash_pct:.2f}%")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 6: Test Buy Conservative Action (Action 1)
print("\n6. Testing Buy Conservative Action (Action 1)...")
try:
    env.reset()
    env.current_step = 50  # Set to a point with enough history
    
    initial_cash = env.cash
    print(f"   Initial cash: ${initial_cash:.2f}")
    
    state, reward, done, info = env.step(1)  # Buy Conservative (25% of cash)
    
    print(f"   ✓ Action executed")
    print(f"   Cash after action: ${env.cash:.2f}")
    print(f"   Cash spent: ${initial_cash - env.cash:.2f}")
    print(f"   Holdings: {env.holdings}")
    print(f"   Portfolio value: ${info['portfolio_value']:.2f}")
    print(f"   Reward: {reward:.6f}")
    
    # Verify we bought stocks
    if len(env.holdings) > 0:
        print(f"   ✓ Successfully bought {len(env.holdings)} stock(s)")
        for ticker, shares in env.holdings.items():
            price = env._get_stock_price(ticker)
            print(f"      - {ticker}: {shares} shares @ ${price:.2f} = ${shares * price:.2f}")
    else:
        print(f"   ⚠ No stocks purchased (might be due to insufficient funds or data)")
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Test Buy Aggressive Action (Action 2)
print("\n7. Testing Buy Aggressive Action (Action 2)...")
try:
    env.reset()
    env.current_step = 50
    
    initial_cash = env.cash
    print(f"   Initial cash: ${initial_cash:.2f}")
    
    state, reward, done, info = env.step(2)  # Buy Aggressive (100% of cash)
    
    print(f"   ✓ Action executed")
    print(f"   Cash after action: ${env.cash:.2f}")
    print(f"   Holdings: {env.holdings}")
    print(f"   Portfolio value: ${info['portfolio_value']:.2f}")
    
    if len(env.holdings) > 0:
        print(f"   ✓ Successfully bought {len(env.holdings)} stock(s)")
    else:
        print(f"   ⚠ No stocks purchased")
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Test Sell Actions
print("\n8. Testing Sell Actions...")
try:
    env.reset()
    env.current_step = 50
    
    # First buy some stocks
    env.step(2)  # Buy Aggressive to get some holdings
    holdings_before = dict(env.holdings)
    cash_before = env.cash
    portfolio_before = calculate_portfolio_value(
        env.holdings, env.cash, test_data, env.current_step
    )
    
    print(f"   Before sell:")
    print(f"      Cash: ${cash_before:.2f}")
    print(f"      Holdings: {holdings_before}")
    print(f"      Portfolio value: ${portfolio_before:.2f}")
    
    if len(holdings_before) > 0:
        # Test Sell Conservative (25%)
        env.current_step += 1  # Advance time
        state, reward, done, info = env.step(3)  # Sell Conservative
        
        print(f"   After Sell Conservative (25%):")
        print(f"      Cash: ${env.cash:.2f}")
        print(f"      Holdings: {env.holdings}")
        print(f"      Portfolio value: ${info['portfolio_value']:.2f}")
        print(f"      Cash increase: ${env.cash - cash_before:.2f}")
        
        # Test Sell Aggressive (100%)
        if len(env.holdings) > 0:
            env.current_step += 1
            state, reward, done, info = env.step(4)  # Sell Aggressive
            
            print(f"   After Sell Aggressive (100%):")
            print(f"      Cash: ${env.cash:.2f}")
            print(f"      Holdings: {env.holdings}")
            print(f"      Portfolio value: ${info['portfolio_value']:.2f}")
            print(f"   ✓ Sell actions executed")
        else:
            print(f"   ⚠ No holdings left to sell")
    else:
        print(f"   ⚠ No holdings to test sell actions")
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 9: Test Hold Action
print("\n9. Testing Hold Action (Action 0)...")
try:
    env.reset()
    env.current_step = 50
    
    initial_cash = env.cash
    initial_holdings = dict(env.holdings)
    
    state, reward, done, info = env.step(0)  # Hold
    
    print(f"   ✓ Hold action executed")
    print(f"   Cash unchanged: ${env.cash:.2f} (was ${initial_cash:.2f})")
    print(f"   Holdings unchanged: {env.holdings} (was {initial_holdings})")
    
    if env.cash == initial_cash and env.holdings == initial_holdings:
        print(f"   ✓ Hold action correctly does nothing")
    else:
        print(f"   ✗ Hold action modified state incorrectly")
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 10: Test get_worst_performing_stock
print("\n10. Testing get_worst_performing_stock()...")
try:
    env.reset()
    env.current_step = 50
    
    # Buy some stocks first
    env.step(2)  # Buy Aggressive
    
    if len(env.holdings) > 0:
        worst_stock = get_worst_performing_stock(
            test_data, env.current_step, env.holdings
        )
        print(f"   ✓ Worst performing stock: {worst_stock}")
        print(f"   ✓ Stock is in holdings: {worst_stock in env.holdings}")
    else:
        print(f"   ⚠ No holdings to test")
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 11: Test multiple steps
print("\n11. Testing multiple steps in sequence...")
try:
    env.reset()
    env.current_step = 50
    
    print(f"   Step 0: Hold")
    env.step(0)
    print(f"      Cash: ${env.cash:.2f}, Holdings: {len(env.holdings)} stocks")
    
    print(f"   Step 1: Buy Conservative (25%)")
    env.current_step += 1
    env.step(1)
    print(f"      Cash: ${env.cash:.2f}, Holdings: {len(env.holdings)} stocks")
    
    print(f"   Step 2: Buy Aggressive (100%)")
    env.current_step += 1
    env.step(2)
    print(f"      Cash: ${env.cash:.2f}, Holdings: {len(env.holdings)} stocks")
    
    print(f"   Step 3: Sell Conservative (25%)")
    env.current_step += 1
    env.step(3)
    print(f"      Cash: ${env.cash:.2f}, Holdings: {len(env.holdings)} stocks")
    
    print(f"   ✓ Multiple steps executed successfully")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("Portfolio management tests completed!")
print("\nNext steps:")
print("  - Verify all actions work as expected")
print("  - Check that commissions are being deducted correctly")
print("  - Ensure portfolio value calculations are accurate")
print("  - Test with real data when available")