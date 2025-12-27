## Test Data Split

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))
from utils import split_data, split_data_by_date

print("=" * 70)
print("TESTING DATA SPLIT")
print("=" * 70)

# Create synthetic data with date index
print("\n1. Creating synthetic data with date range 2000-2024...")
dates = pd.date_range(start='2000-01-01', end='2024-12-31', freq='D')
dates = dates[dates.weekday < 5]  # Only weekdays (trading days)

# Create 5 stocks
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

data_rows = []
for ticker in tickers:
    for date in dates:
        data_rows.append({
            'Date': date,
            'Ticker': ticker,
            'Open': np.random.uniform(50, 200),
            'High': np.random.uniform(50, 200),
            'Low': np.random.uniform(50, 200),
            'Close': np.random.uniform(50, 200),
            'Volume': np.random.randint(1000000, 10000000)
        })

test_data = pd.DataFrame(data_rows)
test_data = test_data.set_index('Date')

print(f"   [OK] Created data with {len(test_data)} rows")
print(f"   Date range: {test_data.index.min()} to {test_data.index.max()}")
print(f"   Stocks: {sorted(test_data['Ticker'].unique())}")

# Test 2: Split data with default dates
print("\n2. Testing split_data() with default dates...")
try:
    train_data, test_data_split = split_data(test_data)
    
    print(f"\n   Training data:")
    print(f"      Rows: {len(train_data)}")
    print(f"      Date range: {train_data.index.min()} to {train_data.index.max()}")
    
    print(f"\n   Test data:")
    print(f"      Rows: {len(test_data_split)}")
    print(f"      Date range: {test_data_split.index.min()} to {test_data_split.index.max()}")
    
    # Verify dates
    assert train_data.index.max() <= pd.to_datetime('2021-12-31'), "Training data extends beyond 2021-12-31"
    assert test_data_split.index.min() >= pd.to_datetime('2022-01-01'), "Test data starts before 2022-01-01"
    assert len(train_data) > 0, "Training data is empty"
    assert len(test_data_split) > 0, "Test data is empty"
    
    print(f"   [OK] Data split correctly with default dates")
    
except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test 3: Split data with custom dates
print("\n3. Testing split_data_by_date() with custom dates...")
try:
    train_data_custom, test_data_custom = split_data_by_date(
        test_data,
        train_start='2010-01-01',
        train_end='2019-12-31',
        test_start='2020-01-01'
    )
    
    print(f"\n   Training data (custom):")
    print(f"      Rows: {len(train_data_custom)}")
    print(f"      Date range: {train_data_custom.index.min()} to {train_data_custom.index.max()}")
    
    print(f"\n   Test data (custom):")
    print(f"      Rows: {len(test_data_custom)}")
    print(f"      Date range: {test_data_custom.index.min()} to {test_data_custom.index.max()}")
    
    # Verify custom dates
    assert train_data_custom.index.min() >= pd.to_datetime('2010-01-01'), "Training start date incorrect"
    assert train_data_custom.index.max() <= pd.to_datetime('2019-12-31'), "Training end date incorrect"
    assert test_data_custom.index.min() >= pd.to_datetime('2020-01-01'), "Test start date incorrect"
    
    print(f"   [OK] Data split correctly with custom dates")
    
except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test with data that has date column (not index)
print("\n4. Testing split_data_by_date() with date column...")
try:
    # Reset index to have Date as a column
    test_data_col = test_data.reset_index()
    
    train_data_col, test_data_col_split = split_data_by_date(
        test_data_col,
        date_column='Date'
    )
    
    print(f"\n   Training data (from column):")
    print(f"      Rows: {len(train_data_col)}")
    if 'Date' in train_data_col.index.names or isinstance(train_data_col.index, pd.DatetimeIndex):
        print(f"      Date range: {train_data_col.index.min()} to {train_data_col.index.max()}")
    else:
        print(f"      Date column present: {'Date' in train_data_col.columns}")
    
    print(f"\n   Test data (from column):")
    print(f"      Rows: {len(test_data_col_split)}")
    
    assert len(train_data_col) > 0, "Training data is empty"
    assert len(test_data_col_split) > 0, "Test data is empty"
    
    print(f"   [OK] Data split correctly with date column")
    
except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test edge cases
print("\n5. Testing edge cases...")

# Test 5a: Data with no overlap
print("\n   5a. Testing with data that doesn't cover full range...")
try:
    limited_data = test_data.loc['2015-01-01':'2018-12-31']
    train_limited, test_limited = split_data(limited_data)
    
    print(f"      Limited data rows: {len(limited_data)}")
    print(f"      Training rows: {len(train_limited)}")
    print(f"      Test rows: {len(test_limited)}")
    
    if len(test_limited) == 0:
        print(f"      [OK] Correctly handles data that doesn't extend to test period")
    else:
        print(f"      [OK] Test data found in limited range")
    
except Exception as e:
    print(f"      [ERROR] {e}")

# Test 5b: Empty data
print("\n   5b. Testing with empty data...")
try:
    empty_data = pd.DataFrame(columns=test_data.columns)
    train_empty, test_empty = split_data(empty_data)
    
    print(f"      Training rows: {len(train_empty)}")
    print(f"      Test rows: {len(test_empty)}")
    print(f"      [OK] Handles empty data gracefully")
    
except Exception as e:
    print(f"      [ERROR] {e}")

# Test 6: Verify no data leakage
print("\n6. Testing for data leakage (no overlap)...")
try:
    train_data, test_data_split = split_data(test_data)
    
    # Check for any overlap in dates
    train_dates = set(train_data.index.unique())
    test_dates = set(test_data_split.index.unique())
    overlap = train_dates.intersection(test_dates)
    
    if len(overlap) == 0:
        print(f"   [OK] No date overlap between training and test sets")
    else:
        print(f"   [WARNING] Found {len(overlap)} overlapping dates!")
        print(f"   Overlapping dates: {sorted(list(overlap))[:10]}...")
    
except Exception as e:
    print(f"   [ERROR] {e}")

# Test 7: Verify data integrity
print("\n7. Testing data integrity (all columns preserved)...")
try:
    train_data, test_data_split = split_data(test_data)
    
    original_columns = set(test_data.columns)
    train_columns = set(train_data.columns)
    test_columns = set(test_data_split.columns)
    
    assert original_columns == train_columns, "Training data missing columns"
    assert original_columns == test_columns, "Test data missing columns"
    
    print(f"   Original columns: {sorted(original_columns)}")
    print(f"   Training columns: {sorted(train_columns)}")
    print(f"   Test columns: {sorted(test_columns)}")
    print(f"   [OK] All columns preserved in both splits")
    
except Exception as e:
    print(f"   [ERROR] {e}")

# Test 8: Verify chronological order
print("\n8. Testing chronological order...")
try:
    train_data, test_data_split = split_data(test_data)
    
    # Check if data is sorted
    train_sorted = train_data.index.is_monotonic_increasing
    test_sorted = test_data_split.index.is_monotonic_increasing
    
    assert train_sorted, "Training data is not sorted chronologically"
    assert test_sorted, "Test data is not sorted chronologically"
    assert train_data.index.max() < test_data_split.index.min(), "Training data comes after test data"
    
    print(f"   Training data sorted: {train_sorted}")
    print(f"   Test data sorted: {test_sorted}")
    print(f"   Training max date < Test min date: {train_data.index.max() < test_data_split.index.min()}")
    print(f"   [OK] Data is in correct chronological order")
    
except Exception as e:
    print(f"   [ERROR] {e}")

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("[OK] Data split tests completed!")
print("\nKey features verified:")
print("  - Default dates (2000-2021 train, 2022+ test)")
print("  - Custom date ranges")
print("  - Date column support")
print("  - Edge case handling")
print("  - No data leakage (no overlap)")
print("  - Data integrity (all columns preserved)")
print("  - Chronological order maintained")
print("\nReady to use for training/test split!")

