"""
    Template_MachineLearning.py
    author: Nicholas S. Bradford
    June 2016
    
    |         before_trading_start    |            at_market_open            |
    Quantopian -> FeedHandler  ->  Strategy     ->     Trader -> Robinhood
             (data)     (feature matrix)   (prediction)     (trade)
"""

import itertools
import numpy as np
import pandas as pd
from scipy import stats 
from sklearn import svm
import talib

N_TRAIN = 1000
N_DATA_PER_DAY = 7 # use for the unit tests
I_CLOSE_PRICE = 2
 
LONG_POSITION = 1.0 
SHORT_POSITION = -1.0
PREDICT_THRESHOLD = 0.5

#==================================================================================================
# Core

def initialize(context):
    """ Set number of training days, trading security, flags, and last buy price."""
    UnitTest().run()
    set_long_only()
    set_max_order_count(1)
    set_commission(commission.PerTrade(cost=0))
    context.security = sid(24) # pick a stock here! sid(24) is AAPL
    set_benchmark(context.security)
    set_commission(commission.PerTrade(cost=0.0))
    context.flag_bought = False
    context.last_buy_price = None
    context.last_price = None
    context.last_prediction = None
    context.accuracy = []
    schedule_function(rebalance,
                      date_rules.every_day(),
                      time_rules.market_open(hours=0, minutes=1))


def before_trading_start(context, data):
    """ Retrain the Strategy every day before trading starts."""
    context.normalized_data = FeedHandler.ingest(context.security, data)
    context.strategy = StrategyManager.train_strategy(context.normalized_data)

        
def handle_data(context, data):
    """ Runs every minute; not implemented."""
    pass


def rebalance(context, data):
    """ Runs once per day to rebalance the portfolio."""
    # make a new prediction
    prediction = context.strategy.predict_prob(context.normalized_data) #context.strategy.predict(context.normalized_data)
    # Go long
    if prediction > 0.5:
        if context.flag_bought:
            print "HOLD @ " + str(data.current(context.security, 'price'))
        else:
            print "BUY  @ " + str(data.current(context.security, 'price'))
        order_for_robinhood(context, context.security, LONG_POSITION)
        context.last_buy_price = data.current(context.security, 'price')
        context.flag_bought = True
    # Go short
    else:
        print "SELL @ " + str(data.current(context.security, 'price')) + " (last bought @ " + str(context.last_buy_price) + ")"
        order_for_robinhood(context, context.security, SHORT_POSITION)
        context.last_buy_price = None
        context.flag_bought = False
    record_results(context, data, prediction)


def order_for_robinhood(context, security, weight, 
                        order_style=None):
    """ Method cloned from template.
    This is a custom order method for this particular algorithm and
        places orders based on:
        1) How much of each position in context.assets we currently hold
        2) How much cash we currently hold
    This means that if you have existing positions (e.g. AAPL),
        your positions in that security will not be taken into
        account when calculating order amounts.
    The portfolio value that we'll be ordering on is labeled 
        `valid_portfolio_value`.
    If you'd like to use a Stop/Limit/Stop-Limit Order please follow the
        following format:
    STOP - order_style = StopOrder(stop_price)
    LIMIT - order_style = LimitOrder(limit_price)
    STOPLIMIT - order_style = StopLimitOrder(limit_price=x, stop_price=y)
    """
    # We use .95 as the cash because all market orders are converted into 
    # limit orders with a 5% buffer. So any market order placed through
    # Robinhood is submitted as a limit order with (last_traded_price * 1.05)
    valid_portfolio_value = context.portfolio.cash * .95

    for s in [context.security]:
        # Calculate dollar amount of each position in context.assets
        # that we currently hold
        if s in context.portfolio.positions:
            position = context.portfolio.positions[s]
            valid_portfolio_value += position.last_sale_price * position.amount
    # Calculate the percent of each security that we want to hold
    percent_to_order = weight - get_percent_held(context,
                                                 security,
                                                 valid_portfolio_value)
    # If within 1% of target weight, ignore.
    if abs(percent_to_order) < .01:
        return
    else:
        # Calculate the dollar value to order for this security
        value_to_order = percent_to_order * valid_portfolio_value
        if order_style:
            return order_value(security, value_to_order, style=order_style)
        else:
            return order_value(security, value_to_order)


def get_percent_held(context, security, portfolio_value):
    """ Method cloned from template.
        This calculates the percentage of each security that we currently hold in the portfolio.
    """
    if security in context.portfolio.positions:
        position = context.portfolio.positions[security]
        value_held = position.last_sale_price * position.amount
        percent_held = value_held/float(portfolio_value)
        return percent_held
    else:
        # If we don't hold any positions, return 0%
        return 0.0

def has_orders(context, data):
    orders = get_open_orders(context.security)
    if orders:
        print "OPEN ORDER FOUND"
        return True
    return False


def record_results(context, data, prediction):
    """ Record results from today. """
    # record prediction accuracy
    if data.current(context.security, 'price') > context.last_price:
        if context.last_prediction > PREDICT_THRESHOLD:
            tmp_accuracy = 1.0
        else:
            tmp_accuracy = 0.0
    else:
        if context.last_prediction > PREDICT_THRESHOLD:
            tmp_accuracy = 0.0
        else:
            tmp_accuracy = 1.0
    context.accuracy.append(tmp_accuracy)
    record(Accuracy=np.mean(context.accuracy))
    if len(context.accuracy) > 9:
        record("10Day Accuracy", np.mean(context.accuracy[-10:]))
    context.last_prediction = prediction
    context.last_price = data.current(context.security, 'price')
    
    x = context.strategy.algo.predict_proba(context.normalized_data)[-1][1]
    record(PROB=x)

#==================================================================================================
    
class FeedHandler(object):

    @staticmethod
    def fetch_sma(security, data):
        """ Fetch SMA."""
        sma_len = 20
        sma_prices = data.history(assets=[security], fields='price', bar_count=N_TRAIN + sma_len, frequency='1d')
        history_sma = talib.SMA(sma_prices[security], timeperiod=sma_len)
        history_sma = np.delete(history_sma, slice(sma_len), axis=0)
        return history_sma


    @staticmethod
    def fetch_rsi(security, data):
        """ Fetch RSI."""
        rsi_len = 14
        rsi_prices = data.history(assets=[security], fields='price', bar_count=N_TRAIN + rsi_len, frequency='1d')
        history_rsi = talib.RSI(rsi_prices[security], timeperiod=rsi_len)
        history_rsi = np.delete(history_rsi, slice(rsi_len), axis=0)
        return history_rsi


    @staticmethod
    def fetch_macd(security, data):
        """ Fetch MACD."""
        macd_len = 33
        macd_prices = data.history(assets=[security], fields='price', bar_count=N_TRAIN + macd_len, frequency='1d')
        macd_raw, signal, hist = talib.MACD(macd_prices[security], fastperiod=12, slowperiod=26, signalperiod=9)
        history_macd = np.subtract(macd_raw, signal) # macd_raw - signal, element-wise
        history_macd = np.delete(history_macd, slice(macd_len), axis=0)
        return history_macd


    @staticmethod
    def ingest(security, data):
        # context.history = data.history(assets=[security], fields=['open', 'high', 'low', 'close', 'volume'], \
        #                                bar_count=N_TRAIN, frequency='1d')
        history_open = data.history(assets=[security], fields='open', bar_count=N_TRAIN, frequency='1d')
        history_high = data.history(assets=[security], fields='high', bar_count=N_TRAIN, frequency='1d')
        history_low = data.history(assets=[security], fields='low', bar_count=N_TRAIN, frequency='1d')
        history_close = data.history(assets=[security], fields='close', bar_count=N_TRAIN, frequency='1d')
        history_volume = data.history(assets=[security], fields='volume', bar_count=N_TRAIN, frequency='1d')
        history_rsi = FeedHandler.fetch_rsi(security, data)
        history_macd = FeedHandler.fetch_macd(security, data)
        history_sma = FeedHandler.fetch_sma(security, data)

        # NaNs don't show up very often, but just in case...
        raw_prices = FeedHandler.normalize_returns(history_open, history_high, history_low, history_close)
        raw_volume = FeedHandler.backfill_NaN(history_volume.as_matrix())
        raw_rsi = FeedHandler.backfill_NaN(history_rsi)
        raw_macd = FeedHandler.backfill_NaN(history_macd)
        raw_sma = FeedHandler.backfill_NaN(history_sma)
    
        normalized_prices = FeedHandler.preprocess_data(raw_prices)
        normalized_volume = FeedHandler.preprocess_data(raw_volume)
        normalized_rsi = np.asarray([np.asarray([x]) for x in FeedHandler.preprocess_data(raw_rsi)])
        normalized_macd = np.asarray([np.asarray([x]) for x in FeedHandler.preprocess_data(raw_macd)])
        normalized_sma = np.asarray([np.asarray([x]) for x in FeedHandler.preprocess_data(raw_sma)])
    
        assert ( N_TRAIN == len(normalized_prices) == len(normalized_volume)
                == len(normalized_rsi) == len(normalized_macd) == len(normalized_sma) )
        normalized_data = np.column_stack(( normalized_prices,
                                            normalized_volume,
                                            normalized_rsi,
                                            normalized_macd,
                                            normalized_sma,
                            ))
        return normalized_data


    @staticmethod
    def backfill_NaN(data, enable_output=True):
        n_detected = 0
        for i, day in enumerate(data):
            if np.isnan(day):
                n_detected += 1
                if enable_output:
                    print "\tWARNING: NaN detected when processing data."
                if i == 0:
                    data[i] = 0.0 # don't want to introduce lookahead bias
                else:
                    data[i] = data[i-1]
        assert n_detected != len(data), "backfill_NaN found a feature of only NaN."
        return data


    @staticmethod
    def preprocess_data(stock_data):
        """ Takes in data for a single stock."""
        zList = stats.zscore(stock_data, axis=None)
        return zList

    
    @staticmethod
    def normalize_returns(open_price, high, low, close_price):
        """ Combine and normalize price data by daily returns."""
        raw_data = np.vstack((high.as_matrix().flatten(), low.as_matrix().flatten(), close_price.as_matrix().flatten()))
        answer = (raw_data.transpose() - open_price.as_matrix())
        return answer


#==================================================================================================

class StrategyManager(object):

    @staticmethod
    def get_targets(data):
        """ Data stored as (open, high, low, close, volume).
            currently no volume or price.
            Strategy will make purshasing decisions at market open. Thus:
            If TOMORROW's OPEN is higher than TODAY's OPEN, BUY.
        """
        target = []
        for i in xrange(len(data)):
            if i == 0:
                continue
            tomorrow = data[i, :]
            today = data[i-1, :]
            t = 0
            if tomorrow[I_CLOSE_PRICE] > today[I_CLOSE_PRICE]:
                t = 1
            target.append(t) # list with one element, one for high, or zero for low
        assert len(data) == len(target) + 1, "ERROR: data and target must have same length."
        for day in data:
            assert len(day) == N_DATA_PER_DAY, "ERROR: day has " + str(len(day)) + " elements instead of " + str(N_DATA_PER_DAY)
        return target

    
    @staticmethod
    def multiday_vector(x, n_days):
        """ Assemble an unraveled feature vector from multiple days. """
        assert len(x) >= n_days
        return [np.array(x[i: i + n_days]).ravel() for i in xrange(len(x) - n_days + 1)]

    
    @staticmethod
    def train_strategy(normalized_data):
        """ Create and train your strategy based on a data set."""
        return Strategy(normalized_data[:-1], StrategyManager.get_targets(normalized_data), N_PREDICT)


#==================================================================================================

class Strategy(object):

    def __init__(self, x, y):
        """ Constructor for the trading strategy.
        Args:
            x: list of input data in the form
                x = [[x1 x2 ... xn],[x1 x2 ... xn],[x1 x2 ... xn]], # stock 1
            y: list of expected data in the form. All values of y must be 0 or 1.
        """
        self.N_PREDICT = N_PREDICT
        self.algo = svm.SVC(C=100, probability=True)  
        training_data = StrategyManager.multiday_vector(x, self.N_PREDICT)
        training_targets = y[self.N_PREDICT - 1:]
        self.algo.fit(training_data, training_targets)
        #record("Training Score", self.algo.score(training_data, training_targets))

    def score(self, x, y):
        """ """
        x = StrategyManager.multiday_vector(x, self.N_PREDICT)
        y = y[self.N_PREDICT - 1:]
        return self.algo.score(x, y)

    def predict(self, x):
        """ Make a prediction of BUY(1) or SELL(0).
        Args:
            x: a single sequence of input of length similar to training data
        Returns:
            Outputs a list of digits between 0 and 1 for all timesteps in the sequence
        """
        predict_data = StrategyManager.multiday_vector(x, self.N_PREDICT)
        ret =  self.algo.predict(predict_data)
        return ret[-1]
    
    def predict_prob(self, x):
        predict_data = StrategyManager.multiday_vector(x, self.N_PREDICT)
        return self.algo.predict_proba(predict_data)[-1][1]

#==================================================================================================
        
class UnitTest(object):
    
    GOOD = [1.0 for i in xrange(N_DATA_PER_DAY)]
    BAD = [-1.0 for i in xrange(N_DATA_PER_DAY)]
    EXTREME_GOOD = [100.0 for i in xrange(N_DATA_PER_DAY)]
    EXTREME_BAD = [-100.0 for i in xrange(N_DATA_PER_DAY)]
    #NEUTRAL = [0.0 for i in xrange(N_DATA_PER_DAY)]
    SIMPLE_DATA = np.array([GOOD, BAD, GOOD, BAD, GOOD, BAD, GOOD, BAD])
    HARDER_DATA = np.array([GOOD, GOOD, BAD, GOOD, BAD, BAD, GOOD, BAD])
    UNEVEN_DATA = [EXTREME_BAD, EXTREME_BAD, EXTREME_BAD, EXTREME_BAD, EXTREME_GOOD]
    SAME_GOOD_DATA = [GOOD for i in xrange(10)]
    #SAME_NEUTRAL_DATA = [NEUTRAL for i in xrange(len(SAME_GOOD_DATA))]


    def test_get_targets(self):
        targets = StrategyManager.get_targets(self.SIMPLE_DATA)
        assert np.array_equal([0, 1, 0, 1, 0, 1, 0], targets)
        targets = StrategyManager.get_targets(self.HARDER_DATA)
        assert np.array_equal([0, 0, 1, 0, 0, 1, 0], targets)


    def test_preprocess_data(self):
        processed = FeedHandler.preprocess_data(np.array(self.UNEVEN_DATA))
        bad = [-0.5 for i in xrange(N_DATA_PER_DAY)]
        good = [2.0 for i in xrange(N_DATA_PER_DAY)]
        answer = [bad, bad, bad, bad, good]
        assert np.array_equal(processed, answer)


    def test_multiday_vector_1(self):
        result = StrategyManager.multiday_vector(x=self.SIMPLE_DATA, n_days=1)
        answer = self.SIMPLE_DATA
        assert np.array_equal(result, answer)


    def test_multiday_vector_4(self):
        result = StrategyManager.multiday_vector(x=self.SIMPLE_DATA, n_days=4)
        answer = [
            [self.GOOD, self.BAD, self.GOOD, self.BAD],
            [self.BAD, self.GOOD, self.BAD, self.GOOD],
            [self.GOOD, self.BAD, self.GOOD, self.BAD],
            [self.BAD, self.GOOD, self.BAD, self.GOOD],
            [self.GOOD, self.BAD, self.GOOD, self.BAD],
        ]
        answer = [np.asarray(x).ravel() for x in answer]
        assert np.array_equal(result, answer)        


    def test_multiday_vector_8(self):
        result = StrategyManager.multiday_vector(x=self.SIMPLE_DATA, n_days=8)
        answer = [np.asarray(self.SIMPLE_DATA).ravel()]
        assert np.array_equal(result, answer)


    def test_backfill_NaN(self):
        result = FeedHandler.backfill_NaN([1, 2, 3, np.nan, 5, 6, np.nan, np.nan, 9], enable_output=False)
        answer = [1, 2, 3, 3, 5, 6, 6, 6, 9]
        assert np.array_equal(result, answer)
        # case where first element is NaN
        result = FeedHandler.backfill_NaN([np.nan, np.nan, 3, 4, np.nan, 6, np.nan, np.nan, 9], enable_output=False)
        answer = [0, 0, 3, 4, 4, 6, 6, 6, 9]
        assert np.array_equal(result, answer)


    def test_normalize_returns(self):
        open_price = pd.DataFrame([90.0, 110.0])
        high = pd.DataFrame([110.0, 120.0])
        low = pd.DataFrame([80.0, 90.0])
        close_price = pd.DataFrame([100.0, 100.0])
        result = FeedHandler.normalize_returns(open_price, high, low, close_price)
        answer = pd.DataFrame([
            [20.0, -10.0, 10.0],
            [10.0, -20.0, -10.0]
        ])
        assert np.array_equal(result, answer)


    def run(self):
        self.test_preprocess_data()
        self.test_get_targets()
        self.test_multiday_vector_1()
        self.test_multiday_vector_4()
        self.test_multiday_vector_8()
        self.test_backfill_NaN()
        self.test_normalize_returns()
        print "All tests passed!"