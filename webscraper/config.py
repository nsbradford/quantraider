"""
    config.py
    Nicholas S. Bradford
    May 2016

"""

#==============================================================================

URL_QUANTOPIAN = "https://www.quantopian.com/signin"
OUTPUT_FILE_NAME = "log.txt"
EMAIL_ID = "user_email"
PASSWORD_ID = "user_password"
BUTTON_ID = "login-button"
ALGOTITLE_ID = "algotitle"
ALGOPAYLOAD_ID = "payload"
# id_list = [
#   ("returns", "livetrading-stats-returns")
#   ("sharpe", "livetrading-stats-sharpe")
#   ("dollarpnl", "livetrading-stats-dollarpnl"),
#   ("longexposure", "livetrading-stats-longexposure"),
#   ("shortexposure", "livetrading-stats-shortexposure"),
#   ("currentcash", "livetrading-stats-currentcash")
# ]
RETURN_ID = "livetrading-stats-returns"
FAILURE_VALUE = "--"

#===============================================================================

URL_ALGO_LIST = [
    "https://www.quantopian.com/live_algorithms/<YOUR_ALGO_HERE>"
]
