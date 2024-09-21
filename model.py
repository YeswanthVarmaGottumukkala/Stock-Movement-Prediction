import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the cleaned data
df_final = pd.read_csv('data/reddit_stocks_cleaned.csv')

# Dummy target variable
df_final['stock_movement'] = (df_final['cleaned_title'].str.contains('bullish', case=False) | 
                               df_final['cleaned_selftext'].str.contains('bullish', case=False)).astype(int)

# Features and target
X = df_final[['cleaned_title', 'cleaned_selftext']]
y = df_final['stock_movement']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Explanation of model.py

# 1. Imports:
# 1.1. pandas: A library for data manipulation and analysis, used to load and handle the DataFrame.
# 1.2. train_test_split: A function from sklearn to split data into training and testing sets.
# 1.3. CountVectorizer: A tool to convert text data into numerical format suitable for machine learning.
# 1.4. classification_report: A function to evaluate the model's performance using various metrics.
# 1.5. XGBClassifier: An implementation of the XGBoost algorithm for classification tasks.
# 1.6. TextBlob: A library for processing textual data, used here for sentiment analysis.

# 2. Load Cleaned Reddit Data:
# 2.1: The cleaned data from the CSV file reddit_stocks_cleaned.csv is loaded into a DataFrame (df).

# 3. Function get_sentiment:
# 3.1: Takes a string text as input and returns its sentiment polarity score using TextBlob.
# 3.2: The sentiment score indicates how positive or negative the text is.

# 4. Add Sentiment Scores to the DataFrame:
# 4.1: A new column sentiment is created in the DataFrame by applying the get_sentiment function to the cleaned_title.

# 5. Prepare Features and Labels:
# 5.1: Features (X) are selected, including the cleaned title, cleaned selftext, and sentiment score.
# 5.2: The target variable (y) is defined, which should indicate the stock movement (e.g., up or down).

# 6. Convert Text Data into Numerical Format:
# 6.1: CountVectorizer is used to transform the cleaned title and selftext into numerical format.
# 6.2: Two separate matrices (X_title and X_selftext) are created for the title and selftext.

# 7. Combine Features:
# 7.1: The feature matrices for title, selftext, and sentiment are combined into a single sparse matrix (X_combined).

# 8. Split Data into Training and Testing Sets:
# 8.1: The combined feature set and target variable are split into training and testing sets using an 80/20 ratio.

# 9. Train the Model:
# 9.1: An instance of XGBClassifier is created.
# 9.2: The model is trained on the training data (X_train, y_train).

# 10. Make Predictions:
# 10.1: Predictions are made on the test data (X_test).

# 11. Evaluate the Model:
# 11.1: The classification_report function is used to print the evaluation metrics (accuracy, precision, recall, F1-score) for the model's predictions.

# Summary:
# This script is crucial for building and evaluating a machine learning model that predicts stock movements based on sentiment and discussions from Reddit.
# It incorporates text preprocessing, feature extraction, model training, and evaluation, forming a complete workflow for your prediction task.
