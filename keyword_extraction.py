import numpy as np
from model import vectorizer

# Input text (to be set dynamically in app.py)
input_text = ""

# TF-IDF scores
tfidf_scores = vectorizer.transform([input_text])
feature_names = np.array(vectorizer.get_feature_names_out())
sorted_indices = np.argsort(tfidf_scores.toarray()[0])[::-1]
top_keywords = feature_names[sorted_indices][:5]