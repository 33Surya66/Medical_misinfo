import tweepy
import json
import requests
from datetime import datetime
from random import choice

# Twitter API credentials
API_KEY = "your_api_key"
API_SECRET_KEY = "your_api_secret"
ACCESS_TOKEN = "your_access_token"
ACCESS_SECRET = "your_access_secret"

# List of medical hashtags to search for
MEDICAL_HASHTAGS = ["#medicine", "#healthcare", "#publichealth", "#medicalnews", "#healthtech"]

# Proxy list (replace with actual working proxy addresses)
PROXIES = [
    "http://username:password@proxy1:port",
    "http://username:password@proxy2:port",
    "http://username:password@proxy3:port"
]

# Setting up proxy
proxy = {"http": choice(PROXIES), "https": choice(PROXIES)}

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# Function to determine sentiment (simplified example using external API)
def get_sentiment(text):
    sentiment_api_url = "https://api.textprocessing.com/sentiment/"
    response = requests.post(sentiment_api_url, data={"text": text}, proxies=proxy)
    if response.status_code == 200:
        return response.json().get("label", "Neutral")
    return "Neutral"

# Extract tweets
extracted_tweets = []
for hashtag in MEDICAL_HASHTAGS:
    for tweet in tweepy.Cursor(api.search_tweets, q=hashtag, lang="en", tweet_mode="extended").items(100):
        tweet_data = {
            "tweet_id": tweet.id_str,
            "username": tweet.user.screen_name,
            "user_id": tweet.user.id_str,
            "date": tweet.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "content": tweet.full_text,
            "likes": tweet.favorite_count,
            "retweets": tweet.retweet_count,
            "replies": 0,  # Twitter API v1.1 does not provide replies count directly
            "hashtags": [hashtag["text"] for hashtag in tweet.entities["hashtags"]],
            "tweet_url": f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id_str}",
            "source": tweet.source,
            "sentiment": get_sentiment(tweet.full_text),
            "location": tweet.user.location if tweet.user.location else "Unknown",
            "verified": tweet.user.verified
        }
        extracted_tweets.append(tweet_data)

# Save to JSON file
with open("medical_tweets.json", "w") as json_file:
    json.dump(extracted_tweets, json_file, indent=4)

print(f"Extracted {len(extracted_tweets)} medical-related tweets successfully.")
