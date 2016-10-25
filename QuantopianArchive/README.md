# QuantopianArchive

## Algorithms

    TODO

## Quantopian Lecture Series

Find the video lectures and IPython notebooks [here](https://www.quantopian.com/lectures).

1. Introduction to Research
  * Gives an overview of the [IPython](https://ipython.org/) environment in Quantopian Research, with samples of how to access market data.
  * Want to get returns easily? Just do:
  
            > data = get_pricing('MSFT', start_date='2012-1-1', end_date='2015-6-1')
            > X = data['price']
            > R = X.pct_change()[1:] # first element is NaN

2. Introduction to Python
  * Very basic overview of Python syntax.
3. Introduction to NumPy
  * 

4. Pandas: DataFrame, Series, loc/iloc slicing, boolean indexing, handling NaN
5. Plotting Data: data structure, histograms, cumulative histograms, scatter plots
7. Variance: Variance/standard deviation, mean absolute deviation, semivariance/semideivation
8. Linear Regression: 
	* Ordinary Least Squares is the objective function
	* F-statistic can be used similar to P-value (if under threshold, do not look at test results for risk of introducing bias)
	* Regression vs. Correlation: both limited to linear models and are measures of covariance. Linear regression also works for multiple independent variables
	* Warning: you can only ever calculate Alpha and Beta over a specific time period, but you need to calculate a confidence interval on its applicability to the future. 
	* "Beta" in finance typically refers to Beta value for Y=a + b * x where x is the S&P500. Quantopian requires competition algorithms to be between -0.3 and 0.3
	* Beta Hedging: Every asset has a Beta to every other asset, so construct a factor model of form: Y=a + b1x1 + b2x2 ...
	* Risk Exposure: an asset's beta value towards another asset. Minimization is key to all of quant finance.
	* Managing risk: diversification, long short equity (a generalization of pairs trading by creating a ranking system), and hedging (short the equivalent Beta value), none of which can ever fully eliminate risk in a portfolio because they're backwards-looking
9. 
