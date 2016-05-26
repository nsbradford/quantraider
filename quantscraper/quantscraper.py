"""
    quantscraper.py
    Nicholas S. Bradford
    April 2016

"""

import argparse
from time import sleep
import getpass

from selenium import webdriver
from bs4 import BeautifulSoup

#==================================================================================================

ALGO_LIST = [
    ("[ALGO_1_NAME_HERE]", "https://www.quantopian.com/live_algorithms/REST_OF_THE_URL"),
    ("[ALGO_2_NAME_HERE]", "https://www.quantopian.com/live_algorithms/REST_OF_THE_URL")
]

URL_QUANTOPIAN = "https://www.quantopian.com/signin"
OUTPUT_FILE_NAME = "log.txt"
EMAIL_ID = "user_email"
PASSWORD_ID = "user_password"
BUTTON_ID = "login-button"
# id_list = [
#   ("sharpe", "livetrading-stats-sharpe")
#   ("dollarpnl", "livetrading-stats-dollarpnl"),
#   ("longexposure", "livetrading-stats-longexposure"),
#   ("shortexposure", "livetrading-stats-shortexposure"),
#   ("currentcash", "livetrading-stats-currentcash"),
#   ("Returns", "livetrading-stats-returns")
# ]

RETURN_ID = "livetrading-stats-returns"
FAILURE_VALUE = "--"

#==================================================================================================

def login(my_username, my_password):
    """ Login to Quantopian."""
    
    print "Logging in..."
    browser = webdriver.Firefox()
    browser.get(URL_QUANTOPIAN)
    sleep(2)
    username = browser.find_element_by_id(EMAIL_ID)
    username.send_keys(my_username)
    password = browser.find_element_by_id(PASSWORD_ID)
    password.send_keys(my_password)
    browser.find_element_by_id(BUTTON_ID).click()
    sleep(1)
    return browser


def fetch_returns(browser, url_list):
    """ Fetch returns from each of the algorithms from Quantopian."""
    
    return_list = []
    for algo, url in url_list:
        unsuccessful = True
        print algo, ": \t",
        while unsuccessful:
            browser.get(url)
            sleep(1)
            html = browser.page_source
            soup = BeautifulSoup(html) #BeautifulSoup(open("scraper.html"))
            num = soup.findAll("div", {"id": RETURN_ID})[0].string
            if num == FAILURE_VALUE:
                print "...\t",
            else:
                print num
                return_list.append((algo, float(num.strip("%"))))
                unsuccessful = False
    return return_list


def output(return_list):
    """ Write to log file, and print rankings."""
    
    print "Writing to log file..."
    output_file = open(OUTPUT_FILE_NAME, "w")
    for algo, returns in return_list:
        output_file.write(str(returns) + "\t")
    output_file.close()

    print "--Rankings--"
    sorted_performance = reversed(sorted(return_list, key=lambda x: x[1]))
    i = 1
    for algo, returns in sorted_performance:
        print i, "\t", algo, ": \t", (returns), "%"
        i += 1


def main():
    """ Primary function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', default="", type=str, help='URL of live algo.')
    args = parser.parse_args()
    url_list = ALGO_LIST
    if args.url != "":
        url_list = [("cmd-line-algo", args.url)]
    
    my_username = raw_input("Username: ")
    my_password = getpass.getpass("Password: ")
    
    browser = login(my_username, my_password)
    return_list = fetch_returns(browser, url_list)
    output(return_list)
    browser.close()


if __name__ == "__main__":
    main()
