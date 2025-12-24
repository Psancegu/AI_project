## Test State Discretization Functions

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))
from utils import (
    discretize_cash_percentage,
    discretize_trend,
    discretize_volatility,
    discretize_performance,
    state_tuple_to_index,
    index_to_state_tuple
)

print("=" * 70)
print("TESTING STATE DISCRETIZATION FUNCTIONS")
print("=" * 70)

# Test 1: Discretize Cash Percentage
print("\n1. Testing discretize_cash_percentage()...")
test_cases_cash = [
    (0, 0, "0% cash (Invested)"),
    (5, 0, "5% cash (Invested)"),
    (9.9, 0, "9.9% cash (Invested)"),
    (10, 1, "10% cash (Balanced)"),
    (30, 1, "30% cash (Balanced)"),
    (50, 1, "50% cash (Balanced)"),
    (50.1, 2, "50.1% cash (Capital)"),
    (75, 2, "75% cash (Capital)"),
    (100, 2, "100% cash (Capital)"),
]

all_passed = True
for cash_pct, expected, description in test_cases_cash:
    result = discretize_cash_percentage(cash_pct)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_passed = False
    print(f"   {status} {description}: {cash_pct}% → {result} (expected {expected})")

# Test 2: Discretize Trend
print("\n2. Testing discretize_trend()...")
test_cases_trend = [
    (-5.0, 0, "Bearish: -5%"),
    (-2.1, 0, "Bearish: -2.1%"),
    (-2.0, 1, "Neutral: -2% (boundary)"),
    (-1.0, 1, "Neutral: -1%"),
    (0.0, 1, "Neutral: 0%"),
    (1.0, 1, "Neutral: +1%"),
    (2.0, 1, "Neutral: +2% (boundary)"),
    (2.1, 2, "Bullish: +2.1%"),
    (5.0, 2, "Bullish: +5%"),
]

for trend_pct, expected, description in test_cases_trend:
    result = discretize_trend(trend_pct)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_passed = False
    print(f"   {status} {description}: {trend_pct}% → {result} (expected {expected})")

# Test 3: Discretize Volatility
print("\n3. Testing discretize_volatility()...")
test_cases_vol = [
    (0.5, 0, "Normal: 0.5 (σ_30d < σ_365d)"),
    (0.9, 0, "Normal: 0.9 (σ_30d < σ_365d)"),
    (1.0, 0, "Normal: 1.0 (σ_30d = σ_365d, boundary)"),
    (1.1, 1, "High Risk: 1.1 (σ_30d > σ_365d)"),
    (2.0, 1, "High Risk: 2.0 (σ_30d > σ_365d)"),
]

for rel_vol, expected, description in test_cases_vol:
    result = discretize_volatility(rel_vol)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_passed = False
    print(f"   {status} {description}: {rel_vol} → {result} (expected {expected})")

# Test 4: Discretize Performance
print("\n4. Testing discretize_performance()...")
test_cases_perf = [
    (-2.0, 0, "Underperforming: -2.0 pp"),
    (-0.6, 0, "Underperforming: -0.6 pp"),
    (-0.5, 1, "Neutral: -0.5 pp (boundary)"),
    (-0.3, 1, "Neutral: -0.3 pp"),
    (0.0, 1, "Neutral: 0.0 pp"),
    (0.3, 1, "Neutral: +0.3 pp"),
    (0.5, 1, "Neutral: +0.5 pp (boundary)"),
    (0.6, 2, "Overperforming: +0.6 pp"),
    (2.0, 2, "Overperforming: +2.0 pp"),
]

for perf_diff, expected, description in test_cases_perf:
    result = discretize_performance(perf_diff)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_passed = False
    print(f"   {status} {description}: {perf_diff} pp → {result} (expected {expected})")

# Test 5: State Tuple to Index
print("\n5. Testing state_tuple_to_index()...")
test_cases_index = [
    ((0, 0, 0, 0), 0, "First state"),
    ((0, 0, 0, 1), 1, "Second state"),
    ((0, 0, 0, 2), 2, "Third state"),
    ((0, 0, 1, 0), 3, "Fourth state"),
    ((0, 1, 0, 0), 6, "Trend=1, others=0"),
    ((1, 0, 0, 0), 18, "Cash=1, others=0"),
    ((2, 2, 1, 2), 53, "Last state (all max)"),
    ((1, 1, 1, 1), 28, "All middle values"),
]

for state_tuple, expected, description in test_cases_index:
    result = state_tuple_to_index(*state_tuple)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_passed = False
    print(f"   {status} {description}: {state_tuple} → index {result} (expected {expected})")

# Test 6: Index to State Tuple (inverse function)
print("\n6. Testing index_to_state_tuple()...")
print("   Testing inverse relationship with state_tuple_to_index()...")
inverse_passed = True
for i in range(54):
    state_tuple = index_to_state_tuple(i)
    back_to_index = state_tuple_to_index(*state_tuple)
    if back_to_index != i:
        inverse_passed = False
        print(f"   ✗ Index {i} → {state_tuple} → {back_to_index} (FAILED)")
        break

if inverse_passed:
    print(f"   ✓ All 54 states correctly map back and forth")
    print(f"   ✓ Inverse function verified for indices 0-53")

# Test 7: Verify all 54 states are unique
print("\n7. Testing uniqueness of all 54 states...")
all_indices = set()
for cash in range(3):
    for trend in range(3):
        for vol in range(2):
            for perf in range(3):
                idx = state_tuple_to_index(cash, trend, vol, perf)
                all_indices.add(idx)

if len(all_indices) == 54 and min(all_indices) == 0 and max(all_indices) == 53:
    print(f"   ✓ All 54 states are unique")
    print(f"   ✓ Index range: 0 to 53")
else:
    print(f"   ✗ Uniqueness test FAILED")
    print(f"   Found {len(all_indices)} unique indices")
    all_passed = False

# Test 8: Edge cases and error handling
print("\n8. Testing error handling...")
error_tests_passed = True

# Test invalid cash_state
try:
    state_tuple_to_index(3, 0, 0, 0)
    print("   ✗ Should have raised ValueError for cash_state=3")
    error_tests_passed = False
except ValueError as e:
    print(f"   ✓ Correctly raised ValueError for invalid cash_state: {e}")

# Test invalid trend_state
try:
    state_tuple_to_index(0, -1, 0, 0)
    print("   ✗ Should have raised ValueError for trend_state=-1")
    error_tests_passed = False
except ValueError as e:
    print(f"   ✓ Correctly raised ValueError for invalid trend_state: {e}")

# Test invalid vol_state
try:
    state_tuple_to_index(0, 0, 2, 0)
    print("   ✗ Should have raised ValueError for vol_state=2")
    error_tests_passed = False
except ValueError as e:
    print(f"   ✓ Correctly raised ValueError for invalid vol_state: {e}")

# Test invalid index
try:
    index_to_state_tuple(54)
    print("   ✗ Should have raised ValueError for index=54")
    error_tests_passed = False
except ValueError as e:
    print(f"   ✓ Correctly raised ValueError for invalid index: {e}")

# Test negative index
try:
    index_to_state_tuple(-1)
    print("   ✗ Should have raised ValueError for index=-1")
    error_tests_passed = False
except ValueError as e:
    print(f"   ✓ Correctly raised ValueError for negative index: {e}")

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
if all_passed and inverse_passed and error_tests_passed:
    print("✓ ALL TESTS PASSED!")
    print("\nState discretization functions are working correctly.")
    print("Ready to integrate into the environment.")
else:
    print("✗ SOME TESTS FAILED")
    print("Please review the errors above.")

# Example: Complete workflow
print("\n" + "=" * 70)
print("EXAMPLE: Complete State Discretization Workflow")
print("=" * 70)

# Simulate some real values
cash_pct = 35.5
trend_pct = -1.5
rel_vol = 1.2
perf_diff = 0.8

print(f"\nInput values:")
print(f"  Cash percentage: {cash_pct}%")
print(f"  Trend indicator: {trend_pct}%")
print(f"  Relative volatility: {rel_vol}")
print(f"  Performance difference: {perf_diff} pp")

# Discretize
cash_state = discretize_cash_percentage(cash_pct)
trend_state = discretize_trend(trend_pct)
vol_state = discretize_volatility(rel_vol)
perf_state = discretize_performance(perf_diff)

print(f"\nDiscretized states:")
print(f"  Cash state: {cash_state} ({'Invested' if cash_state == 0 else 'Balanced' if cash_state == 1 else 'Capital'})")
print(f"  Trend state: {trend_state} ({'Bearish' if trend_state == 0 else 'Neutral' if trend_state == 1 else 'Bullish'})")
print(f"  Volatility state: {vol_state} ({'Normal' if vol_state == 0 else 'High Risk'})")
print(f"  Performance state: {perf_state} ({'Underperforming' if perf_state == 0 else 'Neutral' if perf_state == 1 else 'Overperforming'})")

# Convert to index
state_tuple = (cash_state, trend_state, vol_state, perf_state)
index = state_tuple_to_index(*state_tuple)

print(f"\nState tuple: {state_tuple}")
print(f"Q-table index: {index}")

# Convert back
recovered_tuple = index_to_state_tuple(index)
print(f"Recovered tuple: {recovered_tuple}")
print(f"Match: {'✓' if recovered_tuple == state_tuple else '✗'}")