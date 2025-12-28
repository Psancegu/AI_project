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
        The start date of the simulation (e.g., "2000-01-01").
        
    current_date : pandas.Timestamp
        The specific date corresponding to the current step in the simulation.
        
    dataset : pandas.DataFrame
        Dataframe containing the raw historical price data from the parquet file.
        
    index_dataset : pandas.DataFrame
        Dataframe containing the pre-processed S&P 500 index data with calculated indicators.

    current_step : int
        The current time step (index) of the simulation.
        
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
        self.start_date = "2000-01-01"
        self.current_date = pd.Timestamp(self.start_date)
        
        
        self.dataset = pd.read_parquet('../data/prices_SP500_2000_23122025.parquet')
        self.index_dataset = self.preprocess_SP500(self.dataset)
        self.index_dataset = self.index_dataset[self.index_dataset.index >= self.start_date]
        print(self.index_dataset)

    
        # Numpy optmitzation to avoid the agent taking 3 hours to train
        self.dates_index_array = self.index_dataset.index.to_numpy()
        self.price_array = self.index_dataset['Close'].to_numpy()
        self.sma_array = self.index_dataset['sma_50d'].to_numpy()
        self.std_30d_array = self.index_dataset['std_dev_30d'].to_numpy()
        self.std_365d_array = self.index_dataset['std_dev_365d'].to_numpy()
        self.perf_30d_array = self.index_dataset['perf_30d'].to_numpy()

        self.current_step = 0
        self.max_steps = len(self.dates_index_array) - 1

    
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
    #env.visualize_data()
    print(f"Fecha Inicio: {env.current_date}")


