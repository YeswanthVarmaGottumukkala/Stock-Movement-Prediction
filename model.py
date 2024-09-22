#Code for Prediction Model
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the cleaned data
df_final = pd.read_csv('data/reddit_stocks_cleaned.csv')

# Fill NaN values with an empty string
df_final['cleaned_title'].fillna('', inplace=True)
df_final['cleaned_selftext'].fillna('', inplace=True)

# Add additional feature: Post length
df_final['post_length'] = df_final['cleaned_selftext'].apply(lambda x: len(x.split()))

# Target variable: stock movement based on sentiment
df_final['stock_movement'] = (df_final['cleaned_title'].str.contains('bullish', case=False) | 
                               df_final['cleaned_selftext'].str.contains('bullish', case=False)).astype(int)

# Features and target
X = df_final[['cleaned_title', 'cleaned_selftext', 'post_length']]
y = df_final['stock_movement']

# Vectorization
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_vectorized = vectorizer.fit_transform(X['cleaned_title'] + ' ' + X['cleaned_selftext'])

# Add post_length to the vectorized features
import scipy
X_vectorized = scipy.sparse.hstack((X_vectorized, X[['post_length']]))

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight='balanced'
)

# K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_vectorized, y, cv=kf, scoring='accuracy')
print("Cross-validated Accuracy: {:.2f} ± {:.2f}".format(cv_scores.mean(), cv_scores.std()))

# Train the model on the full training set
model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["No Movement", "Bullish Movement"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



# Explanation of model.py
# 1. Imports:
# 1.1. pandas: A library for data manipulation and analysis, used to handle data frames.
# 1.2. sklearn.model_selection: Functions for splitting datasets and cross-validation.
# 1.3. sklearn.feature_extraction.text: Provides tools for transforming text data into numerical features.
# 1.4. sklearn.ensemble: Contains ensemble methods, including the Random Forest classifier.
# 1.5. sklearn.metrics: Metrics for evaluating model performance.
# 1.6. imblearn.over_sampling: Provides methods for handling imbalanced datasets, specifically SMOTE.

# 2. Load the cleaned data:
# 2.1. The cleaned data is loaded from a CSV file named 'reddit_stocks_cleaned.csv'.

# 3. Data Preprocessing:
# 3.1. Fill NaN values in 'cleaned_title' and 'cleaned_selftext' columns with empty strings to avoid errors during processing.
# 3.2. Add a new feature 'post_length' that calculates the number of words in 'cleaned_selftext'.

# 4. Target variable creation:
# 4.1. Define the target variable 'stock_movement' based on the presence of the word "bullish" in either the title or selftext.
# 4.2. The target variable is encoded as 1 for bullish movement and 0 for no movement.

# 5. Features and target:
# 5.1. Define features (X) as a combination of cleaned title, cleaned selftext, and post length.
# 5.2. The target variable (y) is set as 'stock_movement'.

# 6. Vectorization:
# 6.1. Use TfidfVectorizer to convert the text data into numerical features.
# 6.2. The maximum number of features is set to 3000, and n-grams of 1 and 2 are used for feature extraction.
# 6.3. Combine the vectorized text features with the 'post_length' feature using sparse matrix operations.

# 7. Train-test split:
# 7.1. Split the dataset into training and testing sets with an 80-20 ratio, ensuring stratification to maintain the distribution of the target variable.

# 8. Handle class imbalance:
# 8.1. Apply SMOTE to the training data to address class imbalance by generating synthetic samples of the minority class.

# 9. Random Forest Classifier:
# 9.1. Initialize the Random Forest Classifier with specified parameters, including class weighting to handle imbalanced classes.

# 10. K-Fold Cross-Validation:
# 10.1. Use Stratified K-Fold cross-validation with 5 splits to evaluate model performance.
# 10.2. Calculate and print the mean and standard deviation of cross-validated accuracy scores.

# 11. Model training:
# 11.1. Train the model on the full resampled training dataset using the fit method.

# 12. Make predictions:
# 12.1. Use the trained model to make predictions on the test dataset.

# 13. Evaluation:
# 13.1. Print the accuracy of the model on the test set.
# 13.2. Generate and display the classification report, including precision, recall, and F1-score for each class.
# 13.3. Print the confusion matrix to visualize the performance of the model in terms of true and false positives and negatives.

# Summary:
# This script is crucial for building and evaluating a machine learning model that predicts stock movements based on sentiment and discussions from Reddit.
# It incorporates text preprocessing, feature extraction, model training, and evaluation, forming a complete workflow for your prediction task.
