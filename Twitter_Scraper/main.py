import asyncio
import json
import csv
from configparser import ConfigParser
from twikit import Client, TooManyRequests

QUERY = "Medical"
TWEET_LIMIT = 500  
JSON_FILE = "medical_tweets.json"
CSV_FILE = "medical_tweets.csv"

# Load credentials
config = ConfigParser()
config.read("config.ini")

try:
    username = config.get("X", "username")
    email = config.get("X", "email")
    password = config.get("X", "password")
except Exception as e:
    raise ValueError("Error reading login credentials from config.ini") from e

# Authenticate with X.com (Twitter)
client = Client(language="en-US")

async def fetch_tweets():
    try:
        print("Logging in...")
        await client.login(auth_info_1=username, auth_info_2=email, password=password)
        print("Login successful.")

        all_tweets = []

        while len(all_tweets) < TWEET_LIMIT:
            print(f"Fetching tweets... Total collected: {len(all_tweets)}")

            tweets = await client.search_tweet(QUERY, product="Top")

            if not tweets:
                print("No more tweets found.")
                break  

            for tweet in tweets:
                tweet_data = {
                    "id": tweet.id,
                    "text": tweet.text,
                    "username": getattr(tweet.user, "screen_name", "N/A"),  # Fix for missing username
                    "name": getattr(tweet.user, "name", "N/A"),
                    "date": str(tweet.created_at),
                    "retweets": tweet.retweet_count,
                    "likes": tweet.favorite_count
                }
                all_tweets.append(tweet_data)

                if len(all_tweets) >= TWEET_LIMIT:
                    break  

        # Save to JSON
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(all_tweets, f, indent=4, ensure_ascii=False)

        # Save to CSV
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text", "username", "name", "date", "retweets", "likes"])
            writer.writeheader()
            writer.writerows(all_tweets)

        print(f"✅ Successfully saved {len(all_tweets)} tweets to '{JSON_FILE}' and '{CSV_FILE}'.")

    except TooManyRequests:
        print("Too many requests. Sleeping before retrying...")
        await asyncio.sleep(60)  
    except Exception as e:
        print(f"Error fetching tweets: {e}")

# Run the async function properly
asyncio.run(fetch_tweets())
