from zipline.api import order, record, symbol
import matplotlib.pyplot as plt

def initialize(context):
    pass


def handle_data(context, data):
    order(symbol('AAPL'), 10)
    record(AAPL=data[symbol('AAPL')].price)

"""
def analyze(context, perf):
	ax1 = plt.subplot(211)
	perf.portfolio_value.plot(ax=ax1)
	ax1.set_ylabel('portfolio value')
	ax2 = plt.subplot(212, sharex=ax1)
	perf.AAPL.plot(ax=ax2)
	ax2.set_ylabel('AAPL stock price')
	plt.show()

"""