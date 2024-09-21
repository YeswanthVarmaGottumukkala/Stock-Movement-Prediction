# Stock-Movement-Prediction

## Overview
This project aims to predict stock movements based on discussions from the Reddit platform. By scraping data from the r/stocks subreddit, performing sentiment analysis, and building a machine learning model, we can gain insights into potential stock price trends.

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
  - `xgboost`
  - `scikit-learn`
  - `textblob`
  - `jupyter`

You can install the required libraries by running:
```bash
pip install -r requirements.txt
```
## Setup Instructions
1.Clone the repository:
```bash

git clone https://github.com/YeswanthVarmaGottumukkala/Stock-Movement-Prediction.git
cd Stock-Movement-Prediction
```

2.Update Reddit API credentials in scraper.py:

Replace YOUR_CLIENT_ID, YOUR_CLIENT_SECRET, and YOUR_USER_AGENT with your actual Reddit API credentials.

## Usage
• The scraper.py file scrapes the latest discussions from the r/stocks subreddit.

• The model.py file builds and evaluates a machine learning model using the scraped data.

• The analysis.ipynb notebook contains steps for data preprocessing, sentiment analysis, and model evaluation.

## Model Evaluation
The model's performance is evaluated using metrics such as accuracy, precision, recall, and confusion matrix. Detailed classification reports are generated to assess the effectiveness of the predictions.

## Future Work
• Explore more advanced models (e.g., LSTM, GRU) for time-series predictions.

• Integrate additional data sources, such as historical stock prices and trading volumes.

• Enhance sentiment analysis with more advanced NLP techniques.

## License
This project is licensed under the MIT License.


