import tweepy

# Replace these with your own credentials
consumer_key = 'your-consumer-key'
consumer_secret = 'your-consumer-secret'
access_token = 'your-access-token'
access_token_secret = 'your-access-token-secret'

# Authentication with Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Example: Get tweets from a specific user
username = 'TwitterUsername'  # Replace with the username of the account you want to scrape
tweet_count = 10  # Number of tweets to scrape

# Fetch tweets
tweets = api.user_timeline(screen_name=username, count=tweet_count, tweet_mode='extended')

# Print the tweets
for tweet in tweets:
    print(f"Tweet by @{tweet.user.screen_name}:")
    print(f"Tweet: {tweet.full_text}")
    print(f"Date: {tweet.created_at}")
    print("-" * 50)
