"""
    manager.py 
    author: Nicholas S. Bradford

    Contains helpers.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc

from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pylab as pl

from zipline.utils.factory import load_bars_from_yahoo

#==============================================================================================


class Manager:

    @staticmethod
    def get_SP500():
        """ Return a list of tickers in the S&P 500."""
        stocks_SP500 = np.genfromtxt('manager/constituents.csv', dtype=None, delimiter=',', skiprows=1, usecols=(0))
        return stocks_SP500


    @staticmethod
    def get_DJIA():
        """ Return a list of tickers in the DJIA."""
        stocks_DJIA = ([
            'MMM', 
            'AXP', 
            'AAPL',
            'BA', 
            'CAT', 
            'CVX', 
            #'CSCS', 
            'KO', 
            'DIS', 
            'DD', 
            'XOM',
            'GE',
            #'G',
            'HD',
            'IBM',
            'INTC',
            'JNJ',
            'JPM',
            'MCD',
            'MRK',
            'MSFT',
            'MKE',
            'PFE',
            'PG',
            'TRV',
            'UTX',
            'UNH',
            'VZ',
            #'V',
            'WMT'
        ])
        return stocks_DJIA


    @staticmethod
    def getRawStockDataList(ticker_list, start, end, days):
        """ Returns a list of tickers and list of stock data from Yahoo Finance."""
        new_ticker_list, stock_data_list = [], []
        for ticker in ticker_list:
            try:
                raw_data = Manager.loadTrainingData(ticker, start, end)
                if len(raw_data) == days:
                    new_ticker_list.append(ticker)
                    stock_data_list.append(raw_data)
                else:
                    print "Stock Error:", ticker, "contained", len(raw_data), "instead of 252."
            except IOError:
                print "Stock Error: could not load", ticker, "from Yahoo."  
        return new_ticker_list, stock_data_list


    @staticmethod
    def loadTrainingData(ticker, start, end):
        """ Data stored as (open, high, low, close, volume, price)
            Only take adjusted (open, high, low, close)
        """
        data = load_bars_from_yahoo(stocks=[ticker], start=start, end=end)
        data = Manager.convertPanelToList(data)
        data = ([                             
                    ([  x[0],   # open
                        x[1],   # high  
                        x[2],   # low   
                        x[3],   # close 
                        #x[4],  # volume
                        #x[5],  # price (same as close)
                    ]) for x in data # data stored as (open, high, low, close, volume, price)
        ])
        return data


    @staticmethod
    def convertPanelToList(data):
        """ Convert pandas.Panel --> pandas.DataFrame --> List of Lists """
        answer = data.transpose(2, 1, 0, copy=True).to_frame()
        answer = answer.values.tolist()
        return answer
        

    @staticmethod
    def preprocessData(stock_data, is_normalize=True):
        """ Takes in data for a single stock.
            Z-score the first 4 elements together, and the 5th element separately.
            Note: volume (element 5) is currently being dropped.
        """
        if not is_normalize:
            return stock_data
        else:
            #stock_data = [x[:4] for x in stock_data]
            zList = stats.zscore(stock_data, axis=None)
            return zList

