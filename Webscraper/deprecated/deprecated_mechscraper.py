"""
	THIS IS DEPRECATED
	and does not work, use only for reference.
	
	mechscraper.py 
	Nicholas S. Bradford
	April 2016

	Dependencies:
		mechanize
		BeautifulSoup4 (included in Anaconda)

"""

from bs4 import BeautifulSoup # find elements
from urllib2 import urlopen # parse the doc from the web
import mechanize # necessary for password protection
import cookielib # necessary for password protection
from time import sleep
import sys

url_quantopian = "https://www.quantopian.com/signin"
url_nsbradford_3 = "https://www.quantopian.com/live_algorithms/570259fc34c61fe4c6000875"

username = "nsbradford@wpi.edu"
password = "?"

ids = [
	("id_returns", "livetrading-stats-returns"),
	("id_dollarpnl", "livetrading-stats-dollarpnl"),
	("id_sharpe", "livetrading-stats-sharpe"),
	("id_longexposure", "livetrading-stats-longexposure"),
	("id_shortexposure", "livetrading-stats-shortexposure"),
	("id_currentcash", "livetrading-stats-currentcash")
]


def my_sleep(seconds):
	for i in xrange(seconds):
		sys.stdout.write('.')
		sleep(1)


def main():
	cj = cookielib.CookieJar()
	br = mechanize.Browser()
	br.set_cookiejar(cj)

	# make sure we don't get HTTP Error 403: request disallowed by robots.txt
	br.set_handle_robots(False)
	br.addheaders = [ ( 'User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1' ) ] 
	
	br.open(url_quantopian)

	br.select_form(nr=0)
	br.form['user[email]'] = username
	br.form['user[password]'] = password
	br.submit()
	url = br.open(url_nsbradford_3) # unfortunately, mechanize doesn't wait for JavaScript to load :(
	html = url.read() #html = urlopen(url_nsbradford_3)
	soup = BeautifulSoup(html) #BeautifulSoup(open("scraper.html"))
	
	for key, value in ids:
		num = soup.find("div", {"id": value})
		print key, num


if __name__ == "__main__":
	main()
