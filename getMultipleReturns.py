import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf
yf.pdr_override()

tradingDays = 252
stockList = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

start_date = pd.to_datetime('2014-05-12')
end_date = pd.to_datetime('2019-05-12')


def download_data(stocks):
    data = pdr.get_data_yahoo(stocks, start=start_date, end=end_date)['Adj Close']
    data.columns = stocks
    return data


def show_data(data):
    data.plot(figsize=(10, 5))
    plt.show()


def calculate_returns(data):
    returns = np.log(data / data.shift(1))
    return returns


def plot_daily_returns(returns):
    returns.plot(figsize=(10, 5))
    plt.show()


def show_cov(returns):
    print(returns.cov() * tradingDays)


def show_mean(returns):
    print(returns.mean() * tradingDays)


def initialize_weights():
    weights = np.random.random(len(stockList))
    weights /= np.sum(weights)
    return weights


def calculate_portfolio_return(returns, weights):
    portfolio_return = np.sum(returns.mean() * weights * tradingDays)
    print("Expected portfolio return: ", portfolio_return)


def calculate_portfolio_variance(returns, weights):
    portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * tradingDays, weights)))
    print("Expected variance:", portfolio_variance)


def generate_portfolios(returns, weights):
    num_portfolios = 10000
    p_returns = []
    p_variances = []

    for i in range(num_portfolios):
        weights = np.random.random(len(stockList))
        weights /= np.sum(weights)
        p_returns.append(np.sum(returns.mean() * weights) * tradingDays)
        p_variances.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * tradingDays, weights))))

    p_returns = np.array(p_returns)
    p_variances = np.array(p_variances)
    return p_returns, p_variances


def plot_portfolios(returns, variances):
    plt.figure(figsize=(10,6))
    plt.scatter(variances, returns, c=returns/variances, marker='o')
    plt.grid(True)
    plt.xlabel('Expected variance')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()


def portfolio_summary(returns, weights):
    weights = np.array(weights)
    portfolio_return = np.sum(returns.mean() * weights) * tradingDays
    dot_product = np.dot(weights.T, np.dot(returns.cov() * tradingDays, weights))
    portfolio_volatility = np.sqrt(dot_product)
    return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])


def main():
    data = download_data(stockList)
    show_data(data)
    returns = calculate_returns(data)
    plot_daily_returns(returns)
    show_cov(returns)
    show_mean(returns)
    weights = initialize_weights()
    calculate_portfolio_return(returns, weights)
    calculate_portfolio_variance(returns, weights)
    portfolios = generate_portfolios(returns, weights)
    plot_portfolios(portfolios[0], portfolios[1])
    portfolio_summary(returns, weights)


main()


