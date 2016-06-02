#quantscraper

Web scraper for the [Quantopian](https://www.quantopian.com/) web site, used to facilitate Automated Trading Competition put on by the [WPI Investing Association](http://users.wpi.edu/~investing/).

Ideally, we'd use the much faster [Mechanize](http://wwwsearch.sourceforge.net/mechanize/) library. However, because the data we're scraping is populated by JavaScript, we're forced to use Selenium, which is as slow as Firefox.

####Dependencies:
<b>Selenium WebDriver</b>: Logging into Quantopian and letting the JavaScript populate our desired data. See documentation at http://www.seleniumhq.org/docs/03_webdriver.jsp

<b>BeautifulSoup4</b>: Searching the HTML for our desired data. See documentation at https://www.crummy.com/software/BeautifulSoup/

    $ pip install -U selenium
    $ pip install beautifulsoup4
        
####Usage: 
You'll be prompted for the username and password of the account immediately when starting the script. 
    
    $ quantscraper.py [-h] [-u URL] [-k]

    optional arguments:
      -h, --help         show this help message and exit
      -u URL, --url URL  URL of live algo.
      -k, --is_keep_open  Keep the browser open after scraping.

As an alternative to providing a URL on the command-line, you may edit the source code declaration URL_ALGO_LIST to contain a list of the desired URLs. This is the way to go if you're trying to gather data for a lot of different algorithms. Note that the name string is just for your benefit for when the algos are sorted and ranked at the end of the script; the URL is the important part.
    
    URL_ALGO_LIST = [
      "https://www.quantopian.com/live_algorithms/REST_OF_THE_URL",
      "https://www.quantopian.com/live_algorithms/REST_OF_THE_URL"
    ]
    
####Open for work:
* parallelize scraping with multithreading
* add more complex features such as uploading code, running backtests, etc.
 
