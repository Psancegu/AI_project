#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 09:30:00 2025

@author: Pol
@environment: conda base (Python 3.13.9)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

class AiTrading():
    """
    The class to represent the trading environment
    
    ...

    Attributes:
    -----------
    qtable : numpy.ndarray
        The Q-Learning table with shape (54, 5) storing Q-values for each state-action pair.
        
    state : tuple
        The current state of the environment (Trend, Volatility, Performance, etc.).
        
    cash : float
        The amount of cash currently available in the portfolio.
        
    net_worth : float
        The total value of the portfolio (Cash + Current Value of Holdings).
        
    start_date : str
        The start date of the simulation.
        
    current_date : pandas.Timestamp
        The specific date corresponding to the current step in the simulation.
        
    dataset : pandas.DataFrame
        Dataframe containing the raw historical price data from the parquet file.
        
    index_dataset : pandas.DataFrame
        Dataframe containing the pre-processed S&P 500 index data with calculated metrics.

    current_step : int
        The current time step of the simulation.
        
    max_steps : int
        The total number of steps available in the simulation data.

    dates_index_array : numpy.ndarray
        Optimized array containing the dates for the simulation period.

    price_array : numpy.ndarray
        Optimized array containing the 'Close' prices of the index.

    sma_array : numpy.ndarray
        Optimized array containing the 50-day Simple Moving Average values.

    std_30d_array : numpy.ndarray
        Optimized array containing the 30-day rolling standard deviation (volatility).

    perf_30d_array : numpy.ndarray
        Optimized array containing the 30-day percentage performance of the index.


    Methods:
    --------

    """

    def __init__(self):
        
        self.qtable = np.zeros((54,5))
        self.state = (0,0,0,0) 
        self.cash = 1  
        self.net_worth = 10000
        self.trading_fee = 0.01
        self.start_date = "2000-01-10"
        self.current_date = pd.Timestamp(self.start_date)
        
        self.current_step = 0
        self.stocks = pd.DataFrame(columns=['Quantity', 'Avg_Price'])
        self.stocks.index.name = 'Ticker'
        
        self.portfolio_history = []
        self.portfolio_history.append(self.net_worth)

        self.dataset = pd.read_parquet('../data/prices_SP500_2000_23122025.parquet')
        self.index_dataset = self.preprocess_SP500(self.dataset)
        self.index_dataset = self.index_dataset[self.index_dataset.index >= self.start_date]
        print(self.index_dataset)

    
        self.max_steps = len(self.index_dataset) - 1


    def reset(self):
        """
        Resets the environment to the initial state to start a new training episode.

        Resets cash to initial capital, clears inventory, resets the time step 
        to the start of the training data, and calculates the initial state.

        Returns:
            tuple: The initial state tuple (Cash, Trend, Volatility, Performance).
        """
        self.cash = 10000.0
        self.net_worth = self.cash
        
        self.current_step = 0
        
        self.portfolio_history = []
        self.portfolio_history.append(self.net_worth)

        self.state = self.get_discrete_state()
        
        return self.state

    
    def preprocess_SP500(self, data):
        """
        Calculates all the metrics needed for the problem, from std Deviation to index performance

        Parameters:
            data : pd.DataFrame containing all the data

        Returns:
            pd.DataFrame: A dataframe with all the data calaculated from the index
        """

        # Copy of the data of SP500
        data_SP500 = data[data['Ticker'] == '^GSPC'].copy()

        # Relative daily moving is necessary to calculate std 
        data_SP500['daily_moving_relative'] = data_SP500['Close'].pct_change()

        # Average of the las 50 days, sma_50d
        data_SP500['sma_50d'] = data_SP500['Close'].rolling(window=50).mean()

        # Standard Deviation of the last 30 days
        data_SP500['std_dev_30d'] = data_SP500['daily_moving_relative'].rolling(window=30).std()

        # Standard Deviation of the last year
        data_SP500['std_dev_365d'] = data_SP500['daily_moving_relative'].rolling(window=251).std()

        data_SP500 = data_SP500.fillna(0)

        return data_SP500
    

    def get_discrete_state(self):
        """
        Calculates the discrete state tuple (Cash, Trend, Volatility, Performance) 
        based on the current step in the simulation.

        The state is defined as a tuple of 4 integers:
        1. Cash: (0: <10%, 1: 10-50%, 2: >50%)
        2. Trend: Price vs SMA_50d (0: Bearish, 1: Neutral, 2: Bullish)
        3. Volatility: 30d vs 365d (0: Normal, 1: High Risk)
        4. Performance: Portfolio vs Index 30d (0: Under, 1: Neutral, 2: Over)

        Returns:
            tuple: A tuple of 4 integers representing the current state 
        """
        
        # Component 1: Cash
        cash_ratio = self.cash / self.net_worth
        
        if cash_ratio < 0.10:
            comp1 = 0
        elif cash_ratio <= 0.50:
            comp1 = 1 
        else:
            comp1 = 2

        # Component 2: Trend (Modified to 0-1)
        current_price = self.index_dataset.loc[self.current_date, 'Close']
        current_sma = self.index_dataset.loc[self.current_date, 'sma_50d']
        
        if current_price > current_sma:
            comp2 = 0
        else:
            comp2 = 1

        # Component 3: Relative Volaitily (Modified to 0-1-2) 
        vol_30d = self.index_dataset.loc[self.current_date, 'std_dev_30d']
        vol_365d = self.index_dataset.loc[self.current_date, 'std_dev_365d']

        if abs(vol_30d-vol_365d) < -0.75:
            comp3 = 0
        elif abs(vol_30d-vol_365d) <= 0.75:
            comp3 = 1
        else:
            comp3 = 2  
        
        # Component 4: Portoflio Performance
        minimum_days = 30
        
        if self.current_step < minimum_days:
            comp4 = 1 
        else:
            try:
                prev_nw = self.portfolio_history[self.current_step - minimum_days]
                current_nw = self.net_worth
                
                r_portfolio = (current_nw - prev_nw) / prev_nw
                r_index = self.index_dataset.loc[self.current_date, 'perf_30d']
                
                perf_diff = (r_portfolio - r_index) * 100 

                if perf_diff < -0.5:
                    comp4 = 0 
                elif perf_diff <= 0.5: 
                    comp4 = 1 
                else:
                    comp4 = 2 
                
            except IndexError:
                comp4 = 1

        return (comp1, comp2, comp3, comp4)
    

    def get_state_index(self, state_tuple):
        """
        Maps the state tuple (Cash, Trend, Volatility, Perf) to an index 0-53.
        Formula based on the sizes: Cash(3), Trend(2), Vol(3), Perf(3)
        """
        c, t, v, p = state_tuple
        return (c * 18) + (t * 9) + (v * 3) + p
    

    def buy(self, action):
        """
        Executes the Buy policy.
        """
        trade_executed = False

        if action == 1:
            budget_gross = self.cash * 0.25
        else: 
            budget_gross = self.cash * 1.0
        
        if budget_gross < 10:
            return False

        top_stocks = self.get_info_top5()
        if top_stocks.empty:
            return False
        
        budget_per_stock_gross = budget_gross / len(top_stocks)
    
        for _, row in top_stocks.iterrows():
            ticker = row['Ticker']
            price = row['Close']
            

            budget_net = budget_per_stock_gross / (1 + self.trading_fee)
            
            quantity_to_buy = budget_net / price
            
            if quantity_to_buy > 0:
                cost_of_stocks = quantity_to_buy * price
                commission_cost = cost_of_stocks * self.trading_fee
                
                total_outflow = cost_of_stocks + commission_cost
                
                self.cash -= total_outflow
                
                effective_price = price * (1 + self.trading_fee) 

                if ticker in self.stocks.index:
                    prev_qty = self.stocks.at[ticker, 'Quantity']
                    prev_avg = self.stocks.at[ticker, 'Avg_Price']
                    
                    total_cost_old = prev_qty * prev_avg

                    total_cost_new = quantity_to_buy * effective_price 
                    
                    new_qty = prev_qty + quantity_to_buy
                    new_avg = (total_cost_old + total_cost_new) / new_qty
                    
                    self.stocks.at[ticker, 'Quantity'] = new_qty
                    self.stocks.at[ticker, 'Avg_Price'] = new_avg
                else:
                    new_row = pd.DataFrame(
                        {'Quantity': [quantity_to_buy], 'Avg_Price': [effective_price]}, 
                        index=[ticker]
                    )
                    self.stocks = pd.concat([self.stocks, new_row])
                
                trade_executed = True
        
        self.n_stocks = len(self.stocks)
        return trade_executed
    

    def sell(self, action):
        """
        Executes Sell policy:
        Iteratively sells 100% of the worst performing stocks until 
        the target amount (25% or 100% of portfolio value) is reached.
        """
        if self.stocks.empty:
            return False

        trade_executed = False
        
        current_data = self.dataset.loc[self.current_date]
        
        my_tickers = self.stocks.index.tolist()
        
        my_portfolio_data = current_data[current_data['Ticker'].isin(my_tickers)].copy()


        my_portfolio_data['Quantity'] = my_portfolio_data['Ticker'].map(self.stocks['Quantity'])
    
        my_portfolio_data['Position_Value'] = my_portfolio_data['Quantity'] * my_portfolio_data['Close']
        
        total_holdings_value = my_portfolio_data['Position_Value'].sum()
        
        if action == 3: 
            target_sell_amount = total_holdings_value * 0.25
        else:
            target_sell_amount = total_holdings_value * 1.0

        if target_sell_amount < 1:
            return False

        sorted_portfolio = my_portfolio_data.sort_values(by='perf_30d', ascending=True)
        
        amount_sold_gross = 0.0
        
        for _, row in sorted_portfolio.iterrows():
            if amount_sold_gross >= target_sell_amount:
                break
            
            ticker = row['Ticker']
            position_value = row['Position_Value'] 
            
            cash_in = position_value * (1 - self.trading_fee)
            
            self.cash += cash_in
            
            self.stocks = self.stocks.drop(ticker)

            amount_sold_gross += position_value
            
            trade_executed = True
            
        return trade_executed
        


    def get_info_top5(self):
        """
        Retrieves the data of the top 5 stocks for the current date
        based on the 'rank_30d' column.

        Returns:
            pd.DataFrame: DataFrame containing Ticker, Close, perf_30d, and rank_30d for the top 5.
        """
        daily_data = self.dataset.loc[self.current_date]

        top_5_data = daily_data[daily_data['rank_30d'] <= 5]

        top_5_data = top_5_data.sort_values(by='rank_30d', ascending=True)

        return top_5_data[['Ticker', 'Close', 'perf_30d', 'rank_30d']]

        

    def calculate_reward(self, prev_net_worth, trade_executed):
        """
        Calculates the reward for the current step based on the defined reward function.

        Formula components:
        1. Return (r_t): Adjusted with a 1.5x multiplier for losses (Fear factor).
        2. Commissions: -0.1 penalty if a trade was executed.
        3. Active Reward: Bonus (0.04 or 0.02) based on % of cash deployed to encourage trading.

        Parameters:
            action : int
                The action taken in the step.
            prev_net_worth : float
                The total portfolio value at the beginning of the step.
            trade_executed : bool
                Whether a buy/sell action actually occurred (to apply commission).

        Returns:
            float: The calculated reward value.
        """
        current_net_worth = self.net_worth
        
        if prev_net_worth == 0: 
            rt = 0
        else:
            rt = (current_net_worth - prev_net_worth) / prev_net_worth
        

        if rt < 0:
            reward = 1.5 * rt
        else:
            reward = rt
            
        if trade_executed:
            reward -= 0.1
            
        cash_ratio = self.cash / current_net_worth
        
        if cash_ratio < 0.10:
            reward += 0.04
        elif cash_ratio <= 0.50:
            reward += 0.02
        else:                     
            reward += 0
            
        return reward
    

    def get_current_portfolio_value(self):
        """
        Calculates the total market value of the stocks held.
        Optimized for DataFrame operations.
        """
        if self.stocks.empty:
            return 0.0
            
        current_market_data = self.dataset.loc[self.current_date]

        my_tickers = self.stocks.index
        
        relevant_market = current_market_data[current_market_data['Ticker'].isin(my_tickers)]
        relevant_market = relevant_market.set_index('Ticker')
        
        values = self.stocks['Quantity'] * relevant_market['Close']
        
        return values.sum()



    def step(self, action):
        """
        advances the environment by one time step based on the action taken.

        Process:
        1. Execute the action (Buy/Sell/Hold).
        2. Advance the current_step index.
        3. Update portfolio value based on new market prices.
        4. Calculate the new state and the reward.
        5. Check if the episode is done (end of data or bankruptcy).

        Parameters:
            action : int
                The action index (0-4) to execute.

        Returns:
            tuple: A tuple containing (next_state, reward, done).
                   - next_state (tuple): The new state after the action.
                   - reward (float): The immediate reward received.
                   - done (bool): Whether the episode has ended.
        """
        prev_net_worth = self.net_worth
        
        trade_executed = False
        
        if action == 0:
            trade_executed = False
        elif action == 1:
            trade_executed = self.buy(action)
        elif action == 2:
            trade_executed = self.buy(action)
        elif action == 3:
            trade_executed = self.sell(action)
        elif action == 4:
            trade_executed = self.sell(action)

        self.current_step += 1
        
        if self.current_step >= self.max_steps:
            done = True

            return self.state, 0, done
        
        else:
            done = False

            self.current_date = self.index_dataset.index[self.current_step]

            stock_value = self.get_current_portfolio_value()
            
            self.net_worth = self.cash + stock_value
            self.portfolio_history.append(self.net_worth)
            

            if self.net_worth < 50:
                done = True
                return self.state, -10, done 

            reward = self.calculate_reward(prev_net_worth, trade_executed)
            
            self.state = self.get_discrete_state()
            
            return self.state, reward, done
    
    def train_single_episode(self):
        """
        Executes one episode training
        """
        train_end_date = pd.Timestamp("2021-12-31")
        
        train_data = self.index_dataset[self.index_dataset.index <= train_end_date]
        self.max_steps = len(train_data) - 1
        
        print("Starting Tranining 2000-2021")

        LEARNING_RATE = 0.1
        DISCOUNT = 0.95
        EPSILON = 1.0           
        EPSILON_DECAY = 0.9995 
        MIN_EPSILON = 0.05

        state_tuple = self.reset()
        state_idx = self.get_state_index(state_tuple)
        
        done = False
        step_counter = 0

        while not done:
            
            if np.random.random() > EPSILON:
                action = np.argmax(self.qtable[state_idx])
            else:
                action = np.random.randint(0, 5)
            
            next_state_tuple, reward, done = self.step(action)
            
            next_state_idx = self.get_state_index(next_state_tuple)
            
            if not done:
                current_q = self.qtable[state_idx, action]
                max_future_q = np.max(self.qtable[next_state_idx])
                
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                
                self.qtable[state_idx, action] = new_q
            else:
                self.qtable[state_idx, action] = reward

            state_idx = next_state_idx
            step_counter += 1
            
            if EPSILON > MIN_EPSILON:
                EPSILON *= EPSILON_DECAY
            
            if step_counter % 252 == 0:
                current_year = self.current_date.year
                print(f"Year: {current_year} | Net Worth: ${self.net_worth:.2f}")
        print("Training Ended")
        print(self.qtable)
        return self.portfolio_history
    

    def train_multi_episode(self, episodes=10):
        """
        Executes training over multiple episodes (2000-2021).
        Epsilon decays after each episode.
        """
        train_end_date = pd.Timestamp("2021-12-31")
        train_data = self.index_dataset[self.index_dataset.index <= train_end_date]
        
        original_max_steps = self.max_steps
        self.max_steps = len(train_data) - 1
        
        print(f"Starting Training...")

        LEARNING_RATE = 0.1
        DISCOUNT = 0.95
        
        EPSILON = 1.0           
        EPSILON_DECAY = 0.995
        MIN_EPSILON = 0.05

        all_final_net_worths = []

        for episode in trange(episodes):
            
            state_tuple = self.reset()
            state_idx = self.get_state_index(state_tuple)
            
            done = False
            
            while not done:
                if np.random.random() > EPSILON:
                    action = np.argmax(self.qtable[state_idx])
                else:
                    action = np.random.randint(0, 5)
                
                next_state_tuple, reward, done = self.step(action)
                next_state_idx = self.get_state_index(next_state_tuple)

                if not done:
                    current_q = self.qtable[state_idx, action]
                    max_future_q = np.max(self.qtable[next_state_idx])
                    
                    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                    
                    self.qtable[state_idx, action] = new_q
                else:
                    self.qtable[state_idx, action] = reward

                state_idx = next_state_idx
            
            if EPSILON > MIN_EPSILON:
                EPSILON *= EPSILON_DECAY
            
            all_final_net_worths.append(self.net_worth)

            print(f"Episode: {episode} | Net Worth: ${self.net_worth:.2f}")

        print("Training Ended")
        
        self.max_steps = original_max_steps
        
        return all_final_net_worths


    def test(self, filename="q_table_trading.parquet"):
        """
        Runs a test simulation using a saved Q-Table (Parquet format).
        """ 
        try:
            df_qtable = pd.read_parquet(filename)
            self.qtable = df_qtable.values
            print(f"Q-Table loaded from '{filename}'")
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return []

        test_start_date = pd.Timestamp("2022-01-01")

        start_step_index = self.index_dataset.index.searchsorted(test_start_date)

        self.cash = 10000.0
        self.net_worth = 10000.0
        self.stocks = pd.DataFrame(columns=['Quantity', 'Avg_Price'])
        self.stocks.index.name = 'Ticker'
        self.portfolio_history = [self.net_worth]

        self.current_step = start_step_index
        self.current_date = self.index_dataset.index[self.current_step]
        self.max_steps = len(self.index_dataset) - 1

        self.state = self.get_discrete_state()
        state_idx = self.get_state_index(self.state)

        print(f" Starting Test...")
        
        done = False
        
        while not done:
            action = np.argmax(self.qtable[state_idx])
            
            if self.current_step % 50 == 0:
                 print(f"Date: {self.current_date.date()} | Action: {action} | Cash: {self.cash:.2f}")
            
            next_state_tuple, reward, done = self.step(action)
            state_idx = self.get_state_index(next_state_tuple)

        print(f"Final result 2022 onwards: ${self.net_worth:.2f}")
        
        return self.portfolio_history
    

    def visualize_data(self):
        """
        Generates and displays a dual-plot visualization of the S&P 500 index data to verify data.

        1. Price vs. Trend: Comparing the 'Close' price against the 50-day Simple Moving Average (SMA).
        2. std_dev_30d vs std_dev_365d Comparing short-term volatility (30-day Std Dev) against long-term volatility 
        """
        data = self.index_dataset
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        ax1.plot(data.index, data['Close'], label='SP500 Close', color='#1f77b4', alpha=0.6)
        ax1.plot(data.index, data['sma_50d'], label='SMA 50d', color='orange', linewidth=1.5)
        
        ax1.set_title('S&P 500: Price vs Trend (1999-2025)')
        ax1.set_ylabel('Price ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        ax2.plot(data.index, data['std_dev_30d'], label='Short-Term Volatility (30d)', color='green', alpha=0.5, linewidth=1)
        ax2.plot(data.index, data['std_dev_365d'], label='Long-Term Volatility (365d)', color='red', linewidth=2)
        
        ax2.set_title('Volatility (Standard Deviation)')
        ax2.set_ylabel('Std Dev')
        ax2.set_xlabel('Date')
        ax2.legend(loc='upper left')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        start_date = pd.Timestamp(self.start_date)
        ax1.axvline(start_date, color='black', linestyle='--', alpha=0.5)
        ax2.axvline(start_date, color='black', linestyle='--', alpha=0.5, label='Simulation Start')

        plt.tight_layout()
        plt.show()
    


if __name__ == "__main__":    
    env = AiTrading()
    
    print("\nTraining Phase (2000-2021) - Multi Episode")
    training_results = env.train_multi_episode(episodes=10)
    
    q_table_filename = "q_table_trading.parquet"
    df_qtable = pd.DataFrame(env.qtable, columns=[str(i) for i in range(env.qtable.shape[1])])
    df_qtable.to_parquet(q_table_filename)
    print(f"Q-Table saved to '{q_table_filename}'")

    plt.figure(figsize=(12, 6))
    plt.plot(training_results, label='Final Net Worth per Episode', color='blue')
    plt.title("Training (10 Episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Final Net Worth ($)")
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\nTesting Phase (2022-Present)")
    
    test_curve = env.test(q_table_filename)
    
    if test_curve:
        plt.figure(figsize=(12, 6))
        
        test_dates = env.index_dataset.index[-len(test_curve):]
        
        sp500_data = env.index_dataset.loc[test_dates]['Close']
        
        initial_index_price = sp500_data.iloc[0]
        initial_balance = test_curve[0] 
        
        sp500_benchmark = (sp500_data / initial_index_price) * initial_balance

        plt.plot(test_dates, test_curve, label='AI Agent', color='green', linewidth=2)
        plt.plot(test_dates, sp500_benchmark, label='S&P 500 (Benchmark)', color='gray', linestyle='--', alpha=0.7)
        
        plt.axhline(y=initial_balance, color='red', linestyle=':', linewidth=1, label='Initial Capital')
        
        plt.title("Performance Comparison: AI vs Market")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        ai_return = ((test_curve[-1] - initial_balance) / initial_balance) * 100
        sp500_return = ((sp500_benchmark.iloc[-1] - initial_balance) / initial_balance) * 100
        
        print(f"\nFINAL SUMMARY:")
        print(f"AI Return:      {ai_return:.2f}% (${test_curve[-1]:.2f})")
        print(f"S&P 500 Return: {sp500_return:.2f}% (${sp500_benchmark.iloc[-1]:.2f})")
        
        if ai_return > sp500_return:
            print("AI outperformed the market.")
        else:
            print("AI underperformed the market.")

        plt.show()