import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf
yf.pdr_override()

stocks = ['AAPL', 'WMT', 'TSLA', 'GE', "AMZN", 'DE']
noa = len(stocks)

start_date = pd.to_datetime('2001-01-01')
end_date = pd.to_datetime('2018-03-29')


# downloading the data from Yahoo API
def download_data(stocks):
    data = pdr.get_data_yahoo(stocks, start=start_date, end=end_date)['Adj Close']
    data.columns = stocks
    return data


def show_data(data):
    data.plot(figsize=(10, 5))
    plt.show()


def calculate_returns(data):
    returns = np.log(data / data.shift(1))
    return returns;


def plot_daily_returns(returns):
    returns.plot(figsize=(10, 5));
    plt.show()


