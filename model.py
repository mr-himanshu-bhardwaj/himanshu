import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------ Load and Preprocess Dataset ------------------

# Get the absolute path to the current file (model.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Build the full path to the CSV file
file_path = os.path.join(BASE_DIR, "dataset", r"datasets\fake_news_dataset.csv")
# Load the CSV file safely
news = pd.read_csv(file_path, encoding='latin1')

news.fillna('', inplace=True)
news['content'] = news['author'] + ' ' + news['title']
news.drop(['author', 'title'], axis=1, inplace=True)

# Split data
x = news['content']
y = news['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_features = vectorizer.fit_transform(x_train)
x_test_features = vectorizer.transform(x_test)

# Train model
model = LogisticRegression()
model.fit(x_train_features, y_train)

# Accuracy
train_accuracy = accuracy_score(model.predict(x_train_features), y_train)
test_accuracy = accuracy_score(model.predict(x_test_features), y_test)

# Here is the change
