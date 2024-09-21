##Step 1: Scraping Data from Reddit
#Code to Scrape Data

import praw
import pandas as pd
import re

# Reddit API credentials
reddit = praw.Reddit(client_id='YOUR_CLIENT_ID',
                     client_secret='YOUR_CLIENT_SECRET',
                     user_agent='YOUR_USER_AGENT')

def scrape_reddit(subreddit_name, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for submission in subreddit.new(limit=limit):
        posts.append({
            'title': submission.title,
            'selftext': submission.selftext,
            'score': submission.score,
            'created': submission.created_utc,
            'url': submission.url
        })
    return pd.DataFrame(posts)

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

if __name__ == "__main__":
    df_stocks = scrape_reddit('stocks', limit=1000)
    df_stocks['cleaned_title'] = df_stocks['title'].apply(clean_text)
    df_stocks['cleaned_selftext'] = df_stocks['selftext'].apply(clean_text)
    df_stocks.to_csv('data/reddit_stocks_cleaned.csv', index=False)


# Explanation of scraper.py

# 1. Imports:
# 1.1. praw: A Python wrapper for the Reddit API, allowing us to interact with Reddit's data.
# 1.2. pandas: A library for data manipulation and analysis, used to create and handle data frames.
# 1.3. re: A module for regular expression operations, useful for cleaning text data.

# 2. Reddit API Credentials:
# 2.1. This section initializes the Reddit API client with your credentials (client_id, client_secret, and user_agent).
# 2.2. You need to replace these placeholders with your actual Reddit API credentials.

# 3. Function scrape_reddit:
# 3.1. Parameters:
# 3.1.1. subreddit_name: The name of the subreddit to scrape (e.g., 'stocks').
# 3.1.2. limit: The maximum number of posts to retrieve.
# 3.2. This function creates a connection to the specified subreddit and collects the latest posts.
# 3.3. For each submission, it stores relevant information (title, selftext, score, creation time, and URL) in a list of dictionaries.
# 3.4. Finally, it converts the list into a Pandas DataFrame for easy manipulation and analysis.

# 4. Function clean_text:
# 4.1. Parameters: Takes a string text as input.
# 4.2. It removes URLs using a regular expression.
# 4.3. It cleans the text by removing non-alphabetic characters and converting everything to lowercase, making it uniform for further analysis.

# 5. Main Execution Block:
# 5.1. This part of the code runs if the script is executed directly (not imported as a module).
# 5.2. It calls the scrape_reddit function to fetch 1000 posts from the r/stocks subreddit.
# 5.3. The clean_text function is applied to both the title and selftext of each post to prepare the data for analysis.
# 5.4. Finally, the cleaned DataFrame is saved to a CSV file named reddit_stocks_cleaned.csv in the data directory.

# 6. Summary:
# 6.1. This script is essential for gathering raw data from Reddit, which will then be processed and analyzed in later stages of your project.
# 6.2. It provides a straightforward way to obtain discussions related to stock movements, forming the foundation for your sentiment analysis and prediction model.
