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

import config

#==================================================================================================

def login(my_username, my_password):
    """ Login to Quantopian."""

    print "Logging in..."
    browser = webdriver.Firefox()
    browser.get(config.URL_QUANTOPIAN)
    sleep(2)
    username = browser.find_element_by_id(config.EMAIL_ID)
    username.send_keys(my_username)
    password = browser.find_element_by_id(config.PASSWORD_ID)
    password.send_keys(my_password)
    browser.find_element_by_id(config.BUTTON_ID).click()
    sleep(1)
    return browser


def fetch_returns(browser, url_list):
    """ Fetch returns from each of the algorithms from Quantopian."""

    return_list = []
    for url in url_list:
        unsuccessful = True
        while unsuccessful:
            browser.get(url)
            sleep(1)
            html = browser.page_source
            soup = BeautifulSoup(html) #BeautifulSoup(open("scraper.html"))

            title = soup.find("div", {"class": config.ALGOTITLE_ID})
            name = title.find("span", {"class": config.ALGOPAYLOAD_ID}).string
            num = soup.find("div", {"id": config.RETURN_ID}).string

            if num == config.FAILURE_VALUE:
                print "[retry...]\n",
            else:
                print name, ": \t", num
                return_list.append((name, float(num.strip("%"))))
                unsuccessful = False
    return return_list


def output(return_list):
    """ Write to log file, and print rankings."""

    print "Writing to log file..."
    output_file = open(config.OUTPUT_FILE_NAME, "w")
    for algo, returns in return_list:
        output_file.write(str(returns) + "\t")
    output_file.close()

    print "--Rankings--"
    sorted_performance = reversed(sorted(return_list, key=lambda x: x[1]))
    i = 1
    for algo, returns in sorted_performance:
        print i, "\t", algo, ": \t", (returns), "%"
        i += 1


def scrape(url_list, is_keep_open):
    """ Primary scraping function."""

    my_username = raw_input("Username: ")
    my_password = getpass.getpass("Password: ")

    browser = login(my_username, my_password)
    return_list = fetch_returns(browser, url_list)
    output(return_list)

    if not is_keep_open:
        browser.close()

#==================================================================================================

def main():
    """ Gather command-line arguments for scrape()."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', default="", type=str, help='URL of live algo.')
    parser.add_argument('-k', '--is_keep_open', action='store_true',
                        help='Keep the browser open after scraping.')
    args = parser.parse_args()
    url_list = config.URL_ALGO_LIST
    if args.url != "":
        url_list = [args.url]

    scrape( url_list=url_list,
            is_keep_open=args.is_keep_open)


if __name__ == "__main__":
    main()
