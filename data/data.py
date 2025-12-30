import yfinance as yf # package needs to be install "pip install yfinance"
import pandas as pd
import os
import numpy as np

def get_SP500_tickers():
    """
    Retrieves the list of S&P 500 tickers from a local CSV file.
    
    Returns:
        list: A list of string tickers ready for download.
    """

    listCSV = "constituents.csv" 
    
    if not os.path.exists(listCSV):
        print(f"Could not open the file '{listCSV}'.")
        return

    df_csv = pd.read_csv(listCSV)
    


    tickers = df_csv['Symbol'].tolist()

    tickers.append("^GSPC")
    
    # Replacing '.' to '-' for the yifinance package
    tickers = [t.replace('.', '-') for t in tickers]
    
    # We white list WBA because is Delisted form yfinance
    black_list = ['WBA'] 
    tickers = [t for t in tickers if t not in black_list]
    
    return tickers
    

def download_data(tickers):
    """
    Downloads historical market data for a given list of tickers using yfinance.
    Also filters data of the movers (in/out companies) in the SP500
    
    Parameters:
        tickers (list): A list of stock symbols.
        
    Returns:
        pd.DataFrame: A MultiIndex DataFrame containing the raw market data.
    """

    movers_file = "movers.csv"
    movers_dict = {}
    
    movers_df = pd.read_csv(movers_file)
    
    movers_df['Incorporation Date'] = pd.to_datetime(movers_df['Incorporation Date'], dayfirst=True, errors='coerce')
    movers_df['Leaving Date'] = pd.to_datetime(movers_df['Leaving Date'], dayfirst=True, errors='coerce')
    
    for _, row in movers_df.iterrows():
        t = str(row['Ticker']).replace('.', '-') 
        start = row['Incorporation Date']
        end = row['Leaving Date']
        movers_dict[t] = (start, end)

    current_tickers_norm = [t.replace('.', '-') for t in tickers]

    all_tickers = list(set(current_tickers_norm + list(movers_dict.keys())))
    
    print(f"Downloading data of {len(all_tickers)} companies...")

    data = yf.download(
        all_tickers, 
        start="1998-01-01",  
        group_by='ticker',  
        auto_adjust=True,  
        threads=True,
        progress=True
    )

    # These four lines of code are here to calculate de perf_30d before cutting the dataframe
    close_prices = data.xs('Close', axis=1, level=1)
    perf_df = close_prices.pct_change(periods=30)
    perf_df.columns = pd.MultiIndex.from_product([perf_df.columns, ['perf_30d']])
    data = pd.concat([data, perf_df], axis=1).sort_index(axis=1)

    # Filtering by date 
    downloaded_tickers = data.columns.get_level_values(0).unique()

    for ticker in downloaded_tickers:
        if ticker in movers_dict:
            start_date, end_date = movers_dict[ticker]

            mask_invalid = (data.index < start_date) | (data.index > end_date)

            data.loc[mask_invalid, (ticker, slice(None))] = np.nan
    
    data = data.dropna(axis=1, how='all')

    
    
    return data
                

def clean_data(data):
    """
    Processes the raw DataFrame into a clean one in a Parquet file.
    
    Parameters:
        data (pd.DataFrame): The raw data from yfinance.
    """
    data = data.stack(level=0, future_stack=True)
    
    data.index.names = ['Date', 'Ticker']
    data.columns.name = None
    
    
    data = data.drop(columns=['Adj Close', 'Open', 'High', 'Low'], errors='ignore')
    data = data.dropna(subset=['Close'])

    data = data.reset_index()

    data['rank_30d'] = data.groupby('Date')['perf_30d'].rank(ascending=False, method='first')

    data = data.sort_values(by=['Ticker', 'Date'])

    data = data.set_index('Date') 
    
    print(data.head())
    print(data.tail())

    data.to_parquet("prices_SP500_2000_23122025.parquet")
    
    x, y = data.shape
    print(f"The dimensions of the data frame are: {x} rows and {y} columns")


clean_data(download_data(get_SP500_tickers()))

"""

Un comment this to check companies that moved

data = pd.read_parquet("prices_SP500_2000_23122025.parquet")
print(data[data['Ticker'] == 'TSLA'])
print(data[data['Ticker'] == 'PLTR'])
print(data[data['Ticker'] == 'APP'])


"""