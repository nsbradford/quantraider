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
