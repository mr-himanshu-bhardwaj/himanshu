import streamlit as st
import pandas as pd
import numpy as np
import model

# Page config
st.set_page_config(page_title= "📰 Fake News Detector", layout="wide")

# Header
st.markdown("<h1 style='text-align:center;'>📰 Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Detect whether a news snippet is <strong>REAL</strong> or <strong>FAKE</strong></p>", unsafe_allow_html=True)
st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Model Info"])

with tab1:
    st.markdown("### 📝 Enter News Text")
    input_news = st.text_area("Paste a news headline or snippet below:")

    if input_news.strip():
        transformed_input = model.vectorizer.transform([input_news])
        prediction = model.model.predict(transformed_input)[0]
        prediction_proba = model.model.predict_proba(transformed_input)[0]
        confidence = max(prediction_proba) * 100

        if prediction == 0:
            st.markdown(f"<h3 style='color:red;'>🚨 FAKE NEWS DETECTED</h3>", unsafe_allow_html=True)
            st.markdown("😡 Be cautious! This news might be misleading.")
        else:
            st.markdown(f"<h3 style='color:green;'>✅ REAL NEWS DETECTED</h3>", unsafe_allow_html=True)
            st.markdown("😇 This news seems trustworthy.")

        st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.progress(int(confidence))

        # Top keywords
        tfidf_scores = model.vectorizer.transform([input_news])
        feature_names = np.array(model.vectorizer.get_feature_names_out())
        sorted_indices = np.argsort(tfidf_scores.toarray()[0])[::-1]
        top_keywords = feature_names[sorted_indices][:5]
        st.markdown("#### 🔍 Top Keywords in Your Input")
        st.write(", ".join(top_keywords))

with tab2:
    st.markdown("### 📈 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Training Accuracy", f"{model.train_accuracy:.2f}")
    col2.metric("Testing Accuracy", f"{model.test_accuracy:.2f}")

    st.markdown("### ⚙️ Model Details")
    st.write("**Model Used:** Logistic Regression")
    st.write("**Feature Extraction:** TF-IDF on `author + title`")
    st.code("TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; font-size:12px;'>Made with ❤️ using Streamlit & Scikit-learn</p>", unsafe_allow_html=True)