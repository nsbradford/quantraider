#!/usr/bin/python

"""
    runner.py

    See README.md for usage.

    When creating a new class, you must ensure that it:
        -accepts a list of lists of stock data in its constructor (data, target, num_epochs)
        -has a predict() method
        -is added to the global strategy_dict{}

"""

import sys
import time
import argparse
import itertools
import numpy as np

# handling dates
import pytz
from datetime import datetime

# clustering and helpers
from manager.manager import Manager
from sklearn.cluster import KMeans

# neural nets
from strategies.naive_bayes import NaiveBayes
from strategies.svm import SVM
from neuralnets.tradingnet import TradingNet
from neuralnets.deepnet import DeepNet, testNN

# backtesting
from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_bars_from_yahoo
from trader import settings
from trader.trader import initialize, handle_data

# analysis
import matplotlib.pyplot as plt


#==============================================================================================

def getTargets(data):
    """ data stored as (open, high, low, close, volume, price).
        currently no volume or price.
    """
    target = []
    for i in range(len(data)):
        if i == 0 or i == 1:
            continue
        row = data[i]
        t = [0]
        if row[3] > 0: # row[0] if the close is higher than 0, which is the normalized open
            t[0] = 1      
        target.append(t) # list with one element, one for high, or zero for low

    assert len(data) == len(target) + 2, "ERROR: data and target must have same length."
    for day in data:
        assert len(day) == 4, "ERROR: day has " + str(len(day)) + " elements instead of 4."
    return target

#==============================================================================================

def trainStrategy(strategy_class, stock_list, epochs_num, is_graph=False):
    """ Train your strategy based on a list of stocks.
        Return: a strategy object with a predict() function to use in the algorithm.
    """
    print "Train..."
    data = []
    target = []
    for stock in stock_list:
        target.append(getTargets(stock))
        data.append(stock[:-2])         # delete last data entry, because it won't be used

    if is_graph:
        print target
        plt.figure("Training Data")
        for i in range(len(context.normalized_data[0])):
           plt.plot([x[i] for x in context.normalized_data])
        plt.legend(['open', 'high', 'low', 'close', 'volume', 'price'], loc='upper left')
        plt.show()

    strategy = strategy_class(data, target, num_epochs=epochs_num)
    return strategy

#==============================================================================================

def graphClusters(clusters):
    """
        Parameters:
            clusters (list): list of list(cluster) of list(stock) of doubles
            dateList (list): list of dates 
    """
    plt.figure("Clusters")
    for i, cluster in enumerate(clusters):
        for stock in cluster:
            plt.subplot(2,len(clusters)/2, i+1)
            plt.ylabel("Cluster" + str(i))
            plt.plot(stock)
    plt.show()

#==============================================================================================

def graphElbowMethod(stock_data_list):
    print "Calculating for elbow method..."
    inertia = []
    for x in range(1, len(stock_data_list)+1):
        kmeans = KMeans(n_clusters=x)
        kmeans.fit([np.array(x).ravel() for x in stock_data_list])
        inertia.append(kmeans.inertia_)
    plt.figure("Elbow Method")
    plt.plot(inertia)
    plt.show()

#==============================================================================================

def createClusters(ticker_list, stock_data_list, cluster_num):
    kmeans = KMeans(n_clusters=cluster_num)
    kmeans_data = [np.array(x).ravel() for x in stock_data_list]
    kmeans.fit(kmeans_data)
    clusters = [[] for x in range(cluster_num)]
    tickers = [[] for x in range(cluster_num)]
    for label, stock_data in itertools.izip(kmeans.labels_, stock_data_list):
        clusters[label].append(stock_data)
        #have to re-find ticker of the stock, because nparray is not hashable.
        ticker = None
        for i, stock in enumerate(stock_data_list):
            if (stock_data == stock).all():
                ticker = ticker_list[i]
                break
        assert ticker is not None, "Ticker could not be found for a stock."
        tickers[label].append(ticker)
    return tickers, clusters

#==============================================================================================

def analyze(ticker_list, perf_list):
    """ To be called after the backtest. Will produce a .png in the /output/ directory."""
    print "Analyze..."
    for i, perf in enumerate(perf_list):
        fig = plt.figure(i)
        plt.plot([x/perf.portfolio_value[1] for x in perf.portfolio_value[1:]])
        plt.plot([x/perf.BENCH[1] for x in perf.BENCH[1:]])
        plt.xlabel("Time (Days)")
        plt.ylabel("Percent Returns vs. SPY" + str(i))
        plt.legend(['algorithm', 'BENCHMARK'], loc='upper left')
        outputGraph = str(ticker_list[i]) + "_" + str(time.strftime("%Y-%m-%d_%H-%M-%S"))
        plt.savefig("output/" + outputGraph, bbox_inches='tight')
        time.sleep(1)
        plt.close(fig)
    #plt.show()

#==============================================================================================

def run_clusters(strategy_class, clustering_tickers, cluster_num, epochs_num, training_start, 
                    training_end, backtest_start, backtest_end, is_graph, is_elbow):
    """ Run the test given command-line args.
        Cluster.
        For each cluster, train a strategy on that cluster.
        For each stock in that cluster, run a backtest.
        Graph results.
    """
    ticker_list, raw_stock_data_list = Manager.getRawStockDataList(clustering_tickers, training_start, training_end, 252)
    normalized_stock_data_list = [Manager.preprocessData(x) for x in raw_stock_data_list]
    print "\nClustering..."
    tickers, clusters = createClusters(ticker_list, normalized_stock_data_list, cluster_num)
    print "# of stocks:   " + str(len(normalized_stock_data_list))
    print "# of clusters: " + str(len(clusters))
    print ""
    for t, c in itertools.izip(tickers, clusters):
        print "\tCluster: " + str(len(c)), "stocks: ",
        for symbol in t:
            print symbol,
        print ""

    if is_graph:
        graphClusters(clusters)
    if is_elbow:
        graphElbowMethod(normalized_stock_data_list)

    for t, cluster in itertools.izip(tickers, clusters):
        settings.STRATEGY_OBJECT = trainStrategy(strategy_class, cluster, epochs_num)
        for ticker in t:
            print "Cluster:", t
            print "Stock:", ticker
            tmp_ticks, tmp_data = Manager.getRawStockDataList([ticker], training_start, training_end, 252)
            settings.BACKTEST_STOCK = ticker
            settings.PRE_BACKTEST_DATA = tmp_data[0]
            print "Create Algorithm..."
            algo_obj = TradingAlgorithm(initialize=initialize, handle_data=handle_data)
            try:
                backtest_data = load_bars_from_yahoo(stocks=[ticker, 'SPY'], start=backtest_start, end=backtest_end)
                try:
                    perf = algo_obj.run(backtest_data)
                    analyze([ticker], [perf])
                except ValueError, e:
                    print str(e)
            except IOError, e:
                print "Stock Error: could not load", ticker, "from Yahoo."  
        print "Only testing one cluster for now - Done!"
        return


#==============================================================================================

def main():
    """ Allows for switching easily between different tests using a different STRATEGY_CLASS.
        strategy_num: Must pick a STRATEGY_CLASS from the strategy_dict.
    """ 
    strategy_dict = {
        1: TradingNet,
        2: SVM,
        3: NaiveBayes,
        4: DeepNet
    }
    np.random.seed(13)
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--strategy_num", default=1, type=int, choices=[key for key in strategy_dict])
    #parser.add_argument("-t", "--training_time", default=1, type=int, choices=[year for year in range(1,2)]) #TODO
    #parser.add_argument("-b", "--backtest_time", default=1, type=int, choices=[year for year in range(1,2)]) #TODO
    parser.add_argument("-e", "--epochs_num", default=2000, type=int)
    parser.add_argument("-c", "--cluster_num", default=20, type=int, choices=[n for n in range(1, 25)])
    #parser.add_argument("-z", "--is_normalize", action='store_false', help='Turn normalization off.')
    #parser.add_argument("-o", "--is_overfit", action='store_false', help='Perform test over training data.')
    parser.add_argument("-g", "--graph_clusters", action='store_true', help='Graph clusters.')
    parser.add_argument("-l", "--graph_elbow", action='store_true', help='Graph elbow method for clustering.')
    args = parser.parse_args()

    training_start = datetime(2013, 1, 1, 0, 0, 0, 0, pytz.utc)
    training_end = datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc)
    backtest_start = datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc)
    backtest_end = datetime(2014, 3, 1, 0, 0, 0, 0, pytz.utc)
    #IS_NORMALIZE = args.is_normalize
    #IS_OVERFIT = args.is_overfit
    print "Using:", str(strategy_dict[args.strategy_num])
    #print "Train", training_time, "year,", "Test", backtest_time, "year."
    clustering_tickers = Manager.get_SP500() # get_DJIA() or get_SP500()
    run_clusters(   strategy_class=strategy_dict[args.strategy_num],
                    clustering_tickers=clustering_tickers,
                    cluster_num=args.cluster_num,
                    epochs_num=args.epochs_num,
                    #training_time=args.training_time, 
                    #backtest_time=args.backtest_time,
                    training_start=training_start,
                    training_end=training_end, 
                    backtest_start=backtest_start,
                    backtest_end=backtest_end,
                    is_graph=args.graph_clusters,
                    is_elbow=args.graph_elbow,
    )
    sys.exit(0)

#==============================================================================================

if __name__ == "__main__":
    main()
