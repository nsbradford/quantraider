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
3. Introduction to NumPy
    * Introduction to Python
3. Introduction to [NumPy](http://www.numpy.org/): linear algebra and covariance
4. Introduction to [Pandas](http://pandas.pydata.org/): DataFrame, Series, loc/iloc slicing, boolean indexing, handling NaN
5. Plotting Data: data structure, histograms, cumulative histograms, scatter plots
6. Means: Geometric and Harmonic means
7. Variance: Variance/standard deviation, mean absolute deviation, semivariance/semideivation
8. Linear Regression: 
    * Ordinary Least Squares is the objective function
    * F-statistic can be used similar to P-value (if under threshold, do not look at test results for risk of introducing bias)
    * Regression vs. Correlation: both limited to linear models and are measures of covariance. Linear regression also works for multiple independent variables
    * Warning: you can only ever calculate Alpha and Beta over a specific time period, but you need to calculate a confidence interval on its applicability to the future. 
    * "Beta" in finance typically refers to Beta value for Y=a + b * x where x is the S&P500. Quantopian requires competition algorithms to be between -0.3 and 0.3
9. Multiple Linear Regression
    * Often, two independent variables are actually related through a third ("confounding")
    * Requires several assumptions:
        * Independent variable is not random
        * Error variance is constant, normally distributed, and not autocorrelated
        * The relationship is not exactly linear (covariance), otherwise there are multiple equation solutions
    * Assemble model incrementally by adding variables with AIC (Akaike information criterion) or BIC (Bayesian) scores
    * The best models still require some human judgement as to the best factors to include
10. Linear Correlation Analysis 
    * First, it's helpful to understand [Covariance](http://mathworld.wolfram.com/Covariance.html) and the Covariance Matrix (the diagonal is the variance of the variable, and non-diagonals are each a covariance between two variables)
    * The Correlation Coefficient measures the extent to which the relationship between two variables is linear, and is in range (-1, 1). Equation: r= Cov(x,y) / ( std(x)std(y) )
    * Uses: find correlated assets, or construct a portfolio of uncorrelated assets
    * Warning: correlation is by definition linear, so will utterly fail for different models.
11. Example implementation of the Long-Short Cross-Sectional Momentum algorithm
12. Random Variables
    * Continuous or Discrete, and addressed according to their probability distribution (e.g. a die roll produces values 1-6 with equal probability)
    * For each probability distribution describing a random variable, there is a probability density function (PDF) if it's descrete, along with a cumulative distribution function (CDF) for F(x) = P(X < x)
    * Bernoulli random variables are binary, and are described by some number of random trials where the probabilities are constant.
    * In Modern Portfolio Theory, stocks are generally assumed to behave like random variables following a normal distribution (see Central Limit Theorem), with the implication that a linear combination of multiple stocks also results in a normal distribution. 
    * Standard Normal random variable: Z = (X - mean) / stddev
    * Use a Jarque-Bera test to check if returns are normally distributed. 
13. Statistical Moments
    * Skew: Positive/right skew has a long tail to the right (median > mode), negative/left skew has a long tail to the left (median , mode)
    * Kurtosis: measures how "peaked" a distribution is compared to the normal (kurtosis=3). Greater kurtosis means high peak with fat tails, while low kurtosis means broad.
    * Jarque-Bera tests whether or not
14. Confidence Intervals
    * We want to discover the population mean, but we only have a sample mean. Thus, we should report with a confidence interval (2 standard deviations for 95%). This represents a 95% probability that the true mean lies within that interval (but it does not mean that our guess is 95% probable - subtle difference).
15. Hypothesis Testing
    * Create a null hypothesis H0 and alternative hypthesis HA, identify appropriate test statistic and its distribution, and calculate and compare the critical value with the significance level.
    * Z-values and P-values
    * t-distribution: meant to be used with sample mean and sample variance, and is shorter and fatter.
    * Chi-squared distribution: use for single variance tests
    * F-distribution: used for comparing variances
    * Please go to Wikipedia to learn more in-depth.
16. Maximum Likelihood Estimates (MLE)
    * The process of estimating the parameters of a statistical model (normal distribution, etc) given observations. 
    * Use case: fitting asset returns to a normal distribution.
17. Spearman Rank Correlation
    * Used for dealing with data not fitting the linear model
    * When there are delays in correlation: re-run correlation with different lags/shifts added (beware that this is prone to multiple comparison bias). See Bonferroni Correction.
    * Important Use Case: Evaluating a Ranking Model, as in a Long-Short Equity strategy (rank stocks, then buy the highly ranked ones and sell the poorly ranked ones) which should be market neutral. Run Spearman Correlation on (modelScores, futureReturns) to get a correlation constant and P-value.
18. Beta Hedging
    * Every asset has a Beta to every other asset, so construct a factor model of form: Y=a + b1x1 + b2x2 ...
    * Risk Exposure: an asset's beta value towards another asset. Minimization is key to all of quant finance.
    * Managing risk: diversification, long-short equity (a generalization of pairs trading by creating a ranking system), and hedging (short the equivalent Beta value), none of which can ever fully eliminate risk in a portfolio because they're backwards-looking
    * Can use Kalman filters to estimate Beta
19. Beta Hedging Algorithm
20. Leverege
    * Fundamentally is just borrowing money to trade with, so that you can maximize your returns. Specifically, we trade "on margin" by taking a loan out from the broker.
    * Multiplying capital base increases both risk and returns, keeping the Sharpe Ratio the same.
    * When dealing with leverege, must remember to factor in the negative cash flow that results from interest payments on the debt. 
    * Risk-adjusted returns: often defined by the Sharpe Ratio
21. Pairs Trading
    * Cointegration: if two securities are cointegrated, there is some linear combination between them that will vary around a mean according to the same probability distribution. This is the foundation for saying that they are Mean Reverting.
    * Do not confuse cointegration with correlation, as two series can be correlated without actually mean reverting (e.g. two securities which both increase simultaneously with different slopes), or mean reverting without being correlated at all.
    * Multiple comparisons bias: If you run a large number of p-value tests on the universe of all securities, you are bound to find some spurious conclusions (i.e. if p<0.05 is significance test, then 5% of all p-value tests you run will be wrong)
    * To prevent lookahead bias, make sure to compute rolling Betas (using moving averages) instead of computing over the entire history window.
22. Basic Pairs Trading Algorithm
23. Advanced Pairs Trading Algorithm
24. Position Concentration Risk: Why to Diversify
    * Demos how volatility/risk decreases with increased number of uncorrelated assets in the portfolio.
    * According to Modern Portfolio Theory, you can calculate the expected return of the portfolio by summing the individual weights with the means and variances of each asset. The overall risk can then be calculated (expected standard deviation).
25. Autocorrelation and Auto Regressive (AR) Models
    * An AR(p) model is linearly dependent on 'p' previous points in the series, as well as some stochastic (imperfectly predictable) term. 
    * These models have heavy tails and so typically incur more risk than normal distributions.
    * To determine if a series is AR, run correlations of points with their previous points. Try for as few params as possible, especially since the significance decreases the farther back in time you go.
26. The Dangers of Overfitting
    * Potentially the largest recurring problem in quant finance; it is nearly never possible to fully eliminate, only mitigate.
    * All acquired data includes some noise around the true signal, so the question is: did we model the signal or the noise?
    * General Rule: if the number of rules approaches the points in the data set, it is too complex and prone to overfitting. It is much better to explain 60% of data with 2 params than 90% with 10 params.
    * Moving Avg window length optimization: a very common pitfall and case for overfitting. Ideally, you pick a robust window size that is not especially prone to shifting conditions.
    * Information Criterion: a measure of the relative quality of a models complexity vs. performance
27. Instability of Estimates
    * Even measures like mean, variance, Sharpe, etc. have an associated error.
28. Model Misspecification
    * Constantly run exhaustive tests to make sure you underlying assumptions are all valid (e.g. do not assume market returns are normally distributed).
29. Violations of Regression Models
    * Focus on the Residuals (predictions on existing data), which should be normally distributed (test for Heteroskedasticity, which is slightly different)
    * Financial data is almost always autocorrelated, which means that most statistical measurements assuming a normal distribution and no autocorrelation will be wrong.
    * Newey-West is a method for computing variance that accounts for autocorrelation.
    * Anscombe Quartet: sample of 4 graphs with the same mean and variance and regression properties, but are obviously very different.
30. Regression Model Instability
    * seaborn has a nice visualization of the extreme regression lines possible for a dataset within a certain confidence interval
    * constantly run tests to verify current conditions fall within bounds/assumptions of model
31. Integration, Cointegration, and Stationarity
    * Stationary Process: meta params (such as mean and variance) do not change. Stationarity can change too!
    * Wold Theorem and the Order of Integration: Any stationary process is I(0)
    * Cointegration: If all series are I(1) and some linear combination of them is I(0), then the time series is cointegrated. Thus, the result lacks much auto-covariance and is mostly noise, validating our pairs trading idea.
32. Value at Risk (VaR) and Conditional Value at Risk (CVaR)
    * Historical (non-parametric) VaR looks at previous returns distributions, while other VaR might assume a normal distribution.
    * CVaR, also called Expected Shortfall (ES) can be considered superior to pure VaR as it takes into account the shape of the returns distribution.
33. Arbitrage Pricing Theory
    * Based off of linear factor models (a lineaer combination modelling the returns of some asset). We're concerned about: are these factors useful? What % of returns can I predict from these factors? How exposed am I to these factors?
    * In a perfect market, the risk-adjusted returns for all assets would be equal. Because there are inefficiencies in the market, arbitrage takes advantages of these mispricings.
    * Capital Asset Pricing Model: generalizes expected returns in terms of the risk-free rate and the market returns.
    * Provides example of calculating the expected return of an asset: Run regression.linear_model.OLS(asset_column, factor_dataframe).fit(), get the p-value to confirm significance, and then extract model params.
    * Suggestion: normalize different factors by Z-score to prevent strage coefficient values.
    * Can use pd.stats.ols.MovingOLS() to create a visualization of the model over time, very important for ascertaining the true predictive power.
    * Warning: individual assets are typically poor targets for arbitrage nowadays, as they're easy for HFT to snipe immediately. The Long-Short Equity strategy mitigates this by ranking, and spreading out the risk among hundreds or thousands of trades.
    * You can then walk forward the model to try and predict the future!
34. Fundamental Factor Models
    * Create models with factors such as a company's sector, size, expenses, etc.
35. Factor Risk Exposure
    * FMCAR: Factor Marginal Contribution Active Risk. Active returns are the difference (covariance) between your returns and the market (or another factor), then divide by the std dev of those returns to get the risk. 
    * However, keep in mind that even the distribution of exposure risk is not normal and is difficult to model! Remember that Standard Deviation is only valid for normal distributions.
    * ARCH/GARCH model: autoregressive model that is popular for distributions like this
36. Long-Short Equity
    * Versatile and Scalable, market neutrality is built-in
    * High minimum capital capacity (dollars needed for strategy to function) due to high transactions costs and "Friction because of Prices": very easy to have uneven positions because stock prices don't divide evenly with the amount requested by the strategy (especially when you're trying to hold hundreds of positions with only a few hundred thousand dollars in capital).
37. Long-Short Equity Algorithm
38. Ranking Universes by Factors: How to evaluate a ranking system
    * 
43. ARCH, GARCH, and GMM: A primer on volatility forecasting models developed with Andrei Kirilenko.
44. Kalman Filters How to use Kalman filters to get a good signal out of noisy data.
45. Example: Kalman Filter Pairs Trade: An algorithm to go along with Kalman Filters.
47. Case Study: Traditional Value Factor: How to build a long/short value factor.
47. Example: Momentum Algorithm: An algorithm to showcase an implementation of a momentum strategy.