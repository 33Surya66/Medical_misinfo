import praw
from prawcore.exceptions import NotFound, Forbidden, RequestException

def fetch_reddit_posts(subreddit_name, keywords, limit=100):
    """
    Fetch posts from a specific subreddit based on keywords.

    Args:
        subreddit_name (str): The name of the subreddit to search.
        keywords (list): A list of keywords to search for.
        limit (int): The maximum number of posts to fetch per keyword.

    Returns:
        list: A list of dictionaries containing post details.
    """
    # Set up Reddit API credentials
    try:
        reddit = praw.Reddit(
            client_id="kPxjunFxn4RNbjNwTd6Buw",         # Replace with your Reddit app's client ID
            client_secret="Q1e1sNQeZmvGPx2ieRGprbwf3r6hawt", # Replace with your Reddit app's client secret
            user_agent="medics"        # Replace with a unique user agent string
        )

        # Check authentication
        if reddit.read_only:
            print("Reddit instance is read-only. Ensure correct credentials.")
            return []

        print("Authenticated with Reddit API.")
        subreddit = reddit.subreddit(subreddit_name)
        print(f"Searching subreddit: {subreddit.display_name}")

        # Collect posts
        posts = []
        for keyword in keywords:
            print(f"Searching for keyword: '{keyword}'")
            try:
                for submission in subreddit.search(keyword, limit=limit):
                    posts.append({
                        "title": submission.title,
                        "url": submission.url,
                        "created_utc": submission.created_utc,
                        "score": submission.score,
                        "id": submission.id,
                        "num_comments": submission.num_comments
                    })
            except NotFound:
                print(f"Subreddit '{subreddit_name}' not found or restricted.")
            except Forbidden:
                print(f"Access denied to subreddit: '{subreddit_name}' (might be private).")
            except Exception as e:
                print(f"An error occurred while searching for '{keyword}': {e}")
        print(f"Total posts fetched: {len(posts)}")
        return posts

    except RequestException as e:
        print(f"Reddit API request failed: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

# Example usage
if __name__ == "__main__":
    subreddit = "AskReddit"  # Replace with your target subreddit, e.g., "covid19"
    keywords = ["vaccine", "covid"]  # Replace with your desired keywords
    limit = 10  # Set the number of posts per keyword

    subreddit_posts = fetch_reddit_posts(subreddit, keywords, limit)
    print("Fetched Posts:")
    for post in subreddit_posts:
        print(f"- {post['title']} (Score: {post['score']}, Comments: {post['num_comments']})")
