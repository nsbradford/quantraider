"""
	runner.py 
	author: Nicholas S. Bradford

"""

from manager import *

def main():
	myMan = Manager()
	myMan.run("data/SPY_1993_to_2015.csv")


if __name__ == "__main__":
	main()