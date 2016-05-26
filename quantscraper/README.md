#quantscraper

Web scraper for the [Quantopian](https://www.quantopian.com/) web site, used to fascilitate Automated Trading Competition put on by the [WPI Investing Association](http://users.wpi.edu/~investing/).

Ideally, we'd use the much faster [Mechanize](http://wwwsearch.sourceforge.net/mechanize/) library. However, because the data we're scraping is populated by JavaScript, we're forced to use Selenium, which is as slow as Firefox.

####Dependencies:
Selenium WebDriver: Logging into Quantopian and letting the JavaScript populate our desired data. See documentation at http://www.seleniumhq.org/docs/03_webdriver.jsp

    $ pip install -U selenium
BeautifulSoup4: Searching the HTML for our desired data. See documentation at https://www.crummy.com/software/BeautifulSoup/

    $ pip install beautifulsoup4
        
####Usage: 
You'll be prompted for the username and password of the account immediately when starting the script. 
    
    $ quantscraper.py [-h] [-u URL]

    optional arguments:
      -h, --help         show this help message and exit
      -u URL, --url URL  URL of live algo.

As an alternative to providing a URL on the command-line, you may edit the source code declaration ALGO_LIST to contain a list of tuples with the desired URLs. This is the way to go if you're trying to gather data for a lot of different algorithms. Note that the name string is just for your benefit for when the algos are sorted and ranked at the end of the script; the URL is the important part.
    
    ALGO_LIST = [
      ("ALGO_1_NAME_HERE", "https://www.quantopian.com/live_algorithms/REST_OF_THE_URL"),
      ("ALGO_2_NAME_HERE", "https://www.quantopian.com/live_algorithms/REST_OF_THE_URL"),
      ("ALGO_3_NAME_HERE", "https://www.quantopian.com/live_algorithms/REST_OF_THE_URL")
    ]
    
####Open for work:
* add option to locate algorithm by name instead of URL
* add more complex features such as uploading code, running backtests, etc.
