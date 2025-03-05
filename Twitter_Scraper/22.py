from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
import time

# Set up Selenium with Chrome (Headless)
options = Options()
options.headless = True
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# Add random user-agent to avoid detection
ua = UserAgent()
options.add_argument(f"user-agent={ua.random}")

# Proxy configuration (you can rotate proxies here as well)
proxy = get_random_proxy()
options.add_argument(f'--proxy-server={proxy}')

driver = webdriver.Chrome(options=options)

try:
    driver.get("https://twitter.com/search?q=medical%20tweets&src=typed_query")
    time.sleep(3)  # wait for the page to load

    # Example: Grab tweet content
    tweets = driver.find_elements_by_xpath("//div[@data-testid='tweet']")
    for tweet in tweets:
        tweet_text = tweet.text
        print(tweet_text)

finally:
    driver.quit()
