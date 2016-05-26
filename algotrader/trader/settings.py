"""
	settings.py 

	Allows Zipline TradingAlgorithm to reference 'global' variables.
	
"""

# glabal strategy assigned in main(), along with the years to train, years to backtest, and epochs to train
#IS_NORMALIZE
#IS_OVERFIT
STRATEGY_OBJECT = None
PRE_BACKTEST_DATA = None
BACKTEST_STOCK = None