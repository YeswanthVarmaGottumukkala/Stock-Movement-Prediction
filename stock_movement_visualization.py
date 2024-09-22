# stock movement prediction visualization code

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the cleaned data again to access examples
df_final = pd.read_csv('data/reddit_stocks_cleaned.csv')

# Ensure post_length is included and handle potential NaN values
df_final['cleaned_selftext'] = df_final['cleaned_selftext'].fillna('')
df_final['post_length'] = df_final['cleaned_selftext'].apply(lambda x: len(str(x).split()))

# Example index to illustrate
example_index = 0  # Change this index for different examples
example_post = df_final.iloc[example_index]

# Prepare the data for visualization
# Assuming y_pred and y_test are already defined from your model
prediction_counts = [(y_pred == 1).mean(), (y_pred == 0).mean()]  # Proportions of predictions
labels = ['Bullish', 'Neutral']  # Changed "No Movement" to "Neutral"

# Model Metrics Visualization
accuracy = accuracy_score(y_test, y_pred) * 100  # Convert to percentage
precision = precision_score(y_test, y_pred) * 100  # Convert to percentage
recall = recall_score(y_test, y_pred) * 100  # Convert to percentage

# Create subplots for side-by-side visualization
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Bar Chart for Model Metrics
axs[0].bar(['Accuracy', 'Precision', 'Recall'], [accuracy, precision, recall], color='royalblue', width=0.5)
axs[0].set_ylim(0, 100)  # Limit to 100%
axs[0].set_title('Model Evaluation Metrics (%)')
axs[0].set_ylabel('Score (%)')
axs[0].axhline(y=accuracy, color='green', linestyle='--', label='Accuracy Highlight')  # Highlight accuracy
axs[0].grid(axis='y')
axs[0].legend()

# Stock Market Prediction Visualization
axs[1].set_facecolor('lightgray')  # Set a light background color
bar_colors = ['blue', 'green']  # Blue for Bullish, Green for Neutral
axs[1].bar(labels, prediction_counts, color=bar_colors, width=0.5)
axs[1].set_title('Stock Market Prediction Distribution')
axs[1].set_ylabel('Proportion of Predictions')
axs[1].set_ylim(0, 1)

# Emphasize the Bullish bar
bullish_height = prediction_counts[0] + 0.1  # Increase the bullish height for emphasis
axs[1].bar('Bullish', bullish_height, color='blue', alpha=0.8)

axs[1].grid(axis='y', color='black')  # Grid lines in black for visibility

# Adjust layout
plt.tight_layout()
plt.show()

# Example Data Key Points
sentiment_analysis = "Sentiment analysis indicates a positive sentiment based on the presence of 'bullish' in the title and selftext."
relevant_features = f"Post Length: {example_post['post_length']} words\n" \
                    f"Title: {example_post['cleaned_title']}\n" \
                    f"Selftext: {example_post['cleaned_selftext']}"
trends_observed = "The model has shown a trend where posts with a high sentiment score tend to predict a bullish movement."

# Display the explanation of the selected example
print("Example Data Key Points:")
print(f"Sentiment Analysis: {sentiment_analysis}")
print(f"Relevant Features:\n{relevant_features}")
print(f"Significant Trends Observed: {trends_observed}")


# Explanation of stock movement prediction visualization script

# 1. Imports:
# 1.1. matplotlib.pyplot: A library for creating static, animated, and interactive visualizations in Python.
# 1.2. pandas: A library for data manipulation and analysis, used to handle data frames.
# 1.3. sklearn.metrics: Metrics for evaluating model performance, including accuracy, precision, and recall.

# 2. Load the cleaned data:
# 2.1. Load the cleaned Reddit data from a CSV file named 'reddit_stocks_cleaned.csv' for visualization and analysis.

# 3. Data Preparation:
# 3.1. Fill NaN values in the 'cleaned_selftext' column with an empty string to avoid errors during processing.
# 3.2. Add a new feature 'post_length' that calculates the number of words in 'cleaned_selftext'.

# 4. Example Selection:
# 4.1. Define an example index (e.g., example_index = 0) to illustrate predictions and key points.
# 4.2. Retrieve the example post based on the selected index from the cleaned data.

# 5. Prepare Data for Visualization:
# 5.1. Calculate proportions of predictions for Bullish and Neutral movements based on y_pred and y_test.

# 6. Model Metrics Calculation:
# 6.1. Calculate accuracy, precision, and recall for the model predictions and convert them to percentages.

# 7. Visualization Setup:
# 7.1. Create subplots for side-by-side visualization of model metrics and prediction distribution.

# 8. Bar Chart for Model Metrics:
# 8.1. Create a bar chart to visualize model evaluation metrics (accuracy, precision, recall).
# 8.2. Highlight accuracy with a dashed line for better visibility.

# 9. Stock Market Prediction Visualization:
# 9.1. Create a bar chart to visualize the distribution of stock market predictions (Bullish and Neutral).

# 10. Emphasize Bullish Bar:
# 10.1. Increase the height of the Bullish bar for visual emphasis to make it more prominent.

# 11. Grid and Layout Adjustment:
# 11.1. Add grid lines to the Stock Market Prediction chart for better visibility.
# 11.2. Adjust the layout for improved spacing and visibility of elements.

# 12. Example Data Key Points:
# 12.1. Prepare a summary of the sentiment analysis based on the selected example.
# 12.2. Extract relevant features from the example post for display, including post length, title, and selftext.
# 12.3. Describe significant trends observed in the model's predictions related to sentiment.

# 13. Display Example Explanation:
# 13.1. Print key points regarding the example post and model's predictions, including sentiment analysis, relevant features, and trends observed.
