import yfinance as yf # package needs to be install "pip install yfinance"
import pandas as pd
import os

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
    
    Parameters:
        tickers (list): A list of stock symbols.
        
    Returns:
        pd.DataFrame: A MultiIndex DataFrame containing the raw market data.
    """

    data = yf.download(
        tickers, 
        start="2000-01-01",  
        group_by='ticker',  
        auto_adjust=True,  
        threads=True
    )

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
    
    
    data = data.drop(columns=['Adj Close'], errors='ignore')
    data = data.dropna(subset=['Close'])

    data = data.reset_index()
    data = data.set_index('Date') 
    
    print(data.head())

    data.to_parquet("prices_SP500_2000_23122025.parquet")
    
    x, y = data.shape
    print(f"The dimensions of the data frame are: {x} rows and {y} columns")


clean_data(download_data(get_SP500_tickers()))