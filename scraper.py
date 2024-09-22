##Step 1: Scraping Data from Reddit
#Code to Scrape Data

import praw
import pandas as pd
import re
import os

# Create the data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

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
            'url': submission.url,
            'num_comments': submission.num_comments  # Add number of comments as a feature
        })
    return pd.DataFrame(posts)

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

if __name__ == "__main__":
    subreddits = ['stocks', 'wallstreetbets', 'investing', 'StockMarket', 'pennystocks', 'Daytrading']
    all_posts = []

    for subreddit in subreddits:
        df = scrape_reddit(subreddit, limit=2000)  # Increase limit for more data
        all_posts.append(df)

    df_all_stocks = pd.concat(all_posts, ignore_index=True)
    df_all_stocks['cleaned_title'] = df_all_stocks['title'].apply(clean_text)
    df_all_stocks['cleaned_selftext'] = df_all_stocks['selftext'].apply(clean_text)
    df_all_stocks.to_csv('data/reddit_stocks_cleaned.csv', index=False)


# Explanation of scraper.py

# 1. Imports:
# 1.1. praw: A Python wrapper for the Reddit API, allowing us to interact with Reddit's data.
# 1.2. pandas: A library for data manipulation and analysis, used to create and handle data frames.
# 1.3. re: A module for regular expression operations, useful for cleaning text data.
# 1.4. os: A module that provides a way to use operating system-dependent functionality, such as creating directories.

# 2. Create Data Directory:
# 2.1. The script checks if a 'data' directory exists.
# 2.2. If it doesn't exist, it creates the directory to ensure a proper location for saving the cleaned CSV file.

# 3. Reddit API Credentials:
# 3.1. This section initializes the Reddit API client with your actual credentials (client_id, client_secret, and user_agent).
# 3.2. Ensure that these credentials are kept secure and not shared publicly.

# 4. Function scrape_reddit:
# 4.1. Parameters:
# 4.1.1. subreddit_name: The name of the subreddit to scrape (e.g., 'stocks').
# 4.1.2. limit: The maximum number of posts to retrieve.
# 4.2. This function connects to the specified subreddit and collects the latest posts.
# 4.3. For each submission, it stores relevant information (title, selftext, score, creation time, URL, and number of comments) in a list of dictionaries.
# 4.4. Finally, it converts the list into a Pandas DataFrame for easy manipulation and analysis.

# 5. Function clean_text:
# 5.1. Parameters: Takes a string text as input.
# 5.2. It removes URLs using a regular expression.
# 5.3. It cleans the text by removing non-alphabetic characters and converting everything to lowercase, making it uniform for further analysis.

# 6. Main Execution Block:
# 6.1. This part of the code runs if the script is executed directly (not imported as a module).
# 6.2. It defines a list of subreddits to scrape for a broader dataset, enhancing the analysis.
# 6.3. It iterates through each subreddit, calling the scrape_reddit function to fetch up to 2000 posts.
# 6.4. The results from all subreddits are concatenated into a single DataFrame (df_all_stocks).
# 6.5. The clean_text function is applied to both the title and selftext of each post to prepare the data for analysis.
# 6.6. Finally, the cleaned DataFrame is saved to a CSV file named reddit_stocks_cleaned.csv in the data directory.

# 7. Summary:
# 7.1. This script is essential for gathering raw data from multiple Reddit subreddits, which will then be processed and analyzed in later stages of your project.
# 7.2. It provides a comprehensive way to obtain discussions related to stock movements, forming a solid foundation for your sentiment analysis and prediction model.
