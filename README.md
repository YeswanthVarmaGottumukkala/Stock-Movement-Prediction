# Stock-Movement-Prediction

## Overview
This project aims to predict stock movements based on discussions from the Reddit platform. By scraping data from the r/stocks subreddit, performing sentiment analysis, and building a machine learning model, we can gain insights into potential stock price trends.

### ! This is important for understanding the model's performance and functionality!
For further analysis and to see the code in action, you can view my [Google Colab notebook](https://colab.research.google.com/drive/1AVFerHnPKNKTxCwzRFKl3llfdJ2p3n-3?usp=drive_link).

## Table of Contents
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Future Work](#future-work)
- [License](#license)

## Requirements
- Python 3.x
- Libraries:
  - `pandas`
  - `praw`
  - `scikit-learn`
  - `textblob`
  - `jupyter`

You can install the required libraries by running:
```bash
pip install -r requirements.txt
```
## Setup Instructions
1. [Clone the repository](https://github.com/YeswanthVarmaGottumukkala/Stock-Movement-Prediction/archive/refs/heads/main.zip).
   - Download the repository and unzip the folder.

## Reddit API Credentials
Here’s how to get your Reddit API credentials step-by-step:

### Step 1: Create a Reddit Account
- Go to [Reddit](https://www.reddit.com): If you don’t have an account, visit Reddit and create one.

### Step 2: Create an Application
1. **Log In**: Log into your Reddit account.
2. **Access App Preferences**:
   - Go to the [app preferences page](https://www.reddit.com/prefs/apps) (you can find this under your account settings).
3. **Create a New Application**:
   - Click on “Create App” or “Create Another App.”
4. **Fill Out the Form**:
   - **Name**: Enter a name for your application (e.g., "RedditScraper").
   - **App type**: Select “script”.
   - **Description**: This can be left blank.
   - **About URL**: This can also be left blank.
   - **Permissions**: Leave it blank.
   - **Redirect URI**: Set this to `http://localhost:8080`.
   - **Developer**: Leave this blank.

### Step 3: Get Your Credentials
1. **Submit the Form**: Click on the “Create app” button at the bottom of the form.
2. **Locate Your Credentials**:
   - After creating the app, you will see your `client_id` (located just under the app name) and `client_secret` (found in the app details).
   - Copy these values as you'll need them for your script.

### Example of What You Will See
- **client_id**: This is usually a string of 14 characters.
- **client_secret**: This is a longer string used for authentication.
- **user_agent**: This is a string that you create to identify your application (e.g., "my_reddit_scraper by /u/yourusername").


### 2.Update Reddit API credentials in scraper.py:

Replace YOUR_CLIENT_ID, YOUR_CLIENT_SECRET, and YOUR_USER_AGENT with your actual Reddit API credentials.

## Usage
• The scraper.py file scrapes the latest discussions from the r/stocks subreddit.

• The model.py file builds and evaluates a machine learning model using the scraped data.

## Model Evaluation
The model's performance is evaluated using the following metrics:

- **Accuracy**: 97.57%

### Classification Report:
![WhatsApp Image 2024-09-22 at 14 24 35_2eac95bf](https://github.com/user-attachments/assets/13093dac-6d9a-4dea-acf8-fedaaab40dc5)

###  Stock Market Prediction Visualization:
![WhatsApp Image 2024-09-22 at 14 25 19_12b58c6b](https://github.com/user-attachments/assets/540eaaca-2eef-42d1-9e0b-c928b1b7e942)

The model achieves high accuracy for predicting 'No Movement' and performs reasonably well in identifying 'Bullish Movement,' despite a smaller number of samples for that class. Future work will focus on further improving the model's performance, especially in 'Bullish Movement' predictions.


## Future Work
• Explore more advanced models (e.g., LSTM, GRU) for time-series predictions.

• Integrate additional data sources, such as historical stock prices and trading volumes.

• Enhance sentiment analysis with more advanced NLP techniques.

## License
This project is licensed under the MIT License.

