---
layout: post
title: "Text Explorer App: End-to-End ML for Text Classification"
subtitle: "Learn how to preprocess, model, and interpret text data interactively"
cover-img: /assets/img/text-explorer-cover.png
thumbnail-img: /assets/img/text-explorer-main.png
share-img: /assets/img/text-explorer-cover.png
gh-repo: arpithub/arpithub.github.io
gh-badge: [star, fork, follow]
tags: [datascience,ml,text-classification,streamlit,pipeline]
comments: true
---

## Intro
Text data surrounds us — from emails and reviews to tweets and articles. The **Text ML Explorer App** was built to help learners understand how machine learning models can classify text step by step. It’s an interactive Streamlit app that takes you from raw text to predictions, making the workflow transparent and approachable.

---

## Motivation
Most ML tutorials focus on code snippets, but students often struggle to see the *big picture*. The Text Explorer App solves this by providing a guided, modular workflow:

- Inspect raw text datasets  
- Apply preprocessing techniques  
- Train models with minimal setup  
- Visualize results and misclassifications  

It’s designed for teaching — plug‑and‑play, with default datasets and clear navigation.

---

## Features

The app is organized into four clear pages, each representing a stage in the text ML pipeline:

- **Dataset Explorer 📂**  
  Preview datasets, inspect rows/columns, and understand how text and labels are structured. This builds intuition about the data before modeling.


- **Preprocessing 🔍**  
  - Tokenization demo: split sentences into words  
  - Bag of Words vs TF‑IDF vectorization  
  - Step‑by‑step TF‑IDF calculation with worked examples  
  - Vocabulary preview to see which words are included  


- **Model Builder 🤖**  
  - Train Logistic Regression, Naive Bayes, and Support Vector Classifier  
  - Compare accuracy across models  
  - Confusion matrix visualization  
  - Top Features chart showing which words drive spam vs ham predictions  
  

- **Results 📊**  
  - Test new messages against the trained model  
  - See predictions (spam/ham) with probability scores  
  - Word clouds for spam vs ham vocabulary
  - Explanation of confidence levels in predictions 

---

## Code Walkthrough

The Text Explorer App is organized into four Streamlit pages, each representing a stage in the NLP pipeline. Let’s walk through the code highlights.

---

### 1. 📄 Dataset Explorer

This page either loads the default **sms_spam.csv** dataset or lets users upload their own CSV file:

```python
import streamlit as st
import pandas as pd

st.title("📄 Dataset Explorer")

# Load default dataset
default_df = pd.read_csv("sms_spam.csv")
st.write("### Default Dataset Preview")
st.write(default_df.head())

# Option to upload custom dataset
uploaded_file = st.file_uploader("Or upload your own CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df
    st.write("### Uploaded Dataset Preview")
    st.write(df.head())
else:
    st.session_state["df"] = default_df
```

👉 **Teaching note:**  
The default dataset `sms_spam.csv` is a classic binary classification problem — distinguishing spam from ham (non‑spam) messages. It’s widely used in NLP tutorials because:

- The labels are simple and intuitive  
- The text samples are short, making preprocessing fast  
- Students can immediately connect predictions to real‑world scenarios (spam filters)  

By including this dataset out of the box, learners can start experimenting right away. At the same time, the upload option keeps the app flexible, allowing exploration with custom datasets like product reviews, tweets, or news headlines.


### 2. 🔍 Preprocessing
Here, tokenization and vectorization are explained and demonstrated:

```python
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download("punkt", quiet=True)

sentence = st.text_input("Enter a sentence to tokenize:",
                         "This is a sample sentence for tokenization.")
tokens = nltk.word_tokenize(sentence)
st.write("Tokens:", tokens)

bow_vectorizer = CountVectorizer(stop_words="english")
tfidf_vectorizer = TfidfVectorizer(stop_words="english", min_df=2)

bow = bow_vectorizer.fit_transform(df[text_column])
tfidf = tfidf_vectorizer.fit_transform(df[text_column])
```

👉 **Teaching note:** 
Preprocessing is where raw text becomes numbers. By showing tokenization and vectorization side‑by‑side, students see:
- How sentences break down into tokens
- How Bag of Words counts word frequency
- How TF‑IDF balances common vs rare words

This transparency helps learners understand why models make certain predictions.

### 3. 🤖 Model Builder
This page trains multiple models and compares their performance:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Classifier": SVC(probability=True)
}
```

👉 **Teaching note:** 
Students often wonder which model is best. By training Logistic Regression, Naive Bayes, and SVC on the same dataset, they can:
- Compare accuracy directly
- Visualize confusion matrices to see strengths/weaknesses
- Learn that “best” depends on context, not just accuracy

This builds intuition about model selection in NLP.

### 4. 📊 Results
Finally, predictions and word clouds are displayed:

```python
from wordcloud import WordCloud

user_input = st.text_area("Enter a sentence to classify:",
                          "Free entry in 2 a weekly competition to win tickets!")
X_new = tfidf_vectorizer.transform([user_input])
prediction = best_model.predict(X_new)[0]
st.success(f"Prediction: **{prediction}**")
```

👉 **Teaching note:** 
The Results page is where theory meets practice. Students can:

- Test their own sentences and see predictions instantly
- Understand confidence scores in classification
- Visualize spam vs ham vocabulary with word clouds

This makes the abstract math behind NLP tangible and interactive.

---

## Lessons Learned

Building the Text Explorer App reinforced several key principles:

- **Transparency matters.** Showing intermediate steps (like TF‑IDF weights) helps students grasp why models behave the way they do.  
- **Keep it modular.** Each page is self-contained, so learners can focus on one concept at a time.  
- **Visuals are powerful.** Word clouds, confusion matrices, and feature charts make abstract math tangible.  
- **Default datasets reduce friction.** By including `sms_spam.csv`, students can start experimenting immediately without setup hurdles.  
- **Session state is essential.** Sharing data across pages ensures a smooth workflow and avoids repetitive uploads or preprocessing.

---

## Next Steps

Future improvements could make the app even more engaging:

- Add **sentiment analysis datasets** (movie reviews, tweets) to broaden use cases.  
- Integrate **advanced models** like LSTMs or Transformers for deeper exploration.  
- Provide **interactive error analysis**, highlighting misclassified text and explaining why.  
- Offer **export options** for trained models and preprocessing pipelines.  
- Include **pedagogical diagrams** (e.g., “Text → Tokens → Vectors → Model → Prediction”) to visualize the workflow.

---

## Conclusion

The Text Explorer App makes NLP approachable by combining **hands-on demos, clear explanations, and interactive visuals**. Whether you’re teaching, learning, or experimenting, this app provides a transparent window into how text becomes numbers — and how those numbers drive predictions.  

👉 Try it out here: [Text Explorer App](https://text-ml-explorer.streamlit.app/)

---

##### References:

- [Text Explorer App Repository](https://github.com/arpitHub/Text-Data-Explorer)  
- [Streamlit Documentation](https://docs.streamlit.io/)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)  
- [NLTK Documentation](https://www.nltk.org/)  
- [WordCloud Library](https://amueller.github.io/word_cloud/)  
