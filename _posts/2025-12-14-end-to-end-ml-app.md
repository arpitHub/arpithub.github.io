---
layout: post
title: Building an End-to-End Machine Learning App with Streamlit
subtitle: Building an End-to-End Machine Learning App with Streamlit
cover-img: /assets/img/pytest_1.jpg
thumbnail-img: /assets/img/pytest_thumb.png
share-img: /assets/img/pytest_2.jpg
gh-repo: arpithub/arpithub.github.io
gh-badge: [star, fork, follow]
tags: [datascience,ml,streamlit,pipeline]
comments: true
---

Over the past few weeks, I’ve been experimenting with Streamlit to build a lightweight, end-to-end machine learning [application](https://end-to-end-ml-app.streamlit.app/). The goal was simple: make it easy to **explore datasets, run quick EDA, and compare models**, all inside a clean web interface. I only wish I had this app during my Northeastern days; it would have made understanding the full data science modeling process much easier.


This post walks through the design, challenges, and lessons learned while building the app.

---

## Motivation

Most ML workflows start with the same steps:

1. Load a dataset
2. Explore it visually
3. Run some baseline models
4. Compare results

I wanted a tool that could do all of this interactively, without heavy dependencies like PyCaret or LazyPredict. The result is a **Light Mode–only Streamlit app** that stays lightweight but still powerful.

---

## Features

The app is structured into four pages:

- **Dataset Explorer**  
  Choose from built-in sklearn datasets, pydataset collections, or upload your own CSV.  
  Preview rows, check shape, and inspect column types.

- **EDA Dashboard**  
  Quick stats, correlation heatmaps, and full profiling reports (via `ydata-profiling`).  
  This helps spot missing values, distributions, and relationships early.

- **Model Builder (Light Mode)**  
  Compare multiple sklearn models (classification or regression) side by side.  
  Models include Logistic Regression, Random Forest, Gradient Boosting, SVC, KNN, Naive Bayes, Decision Trees, and linear regressors.  
  Results are shown in a leaderboard sorted by accuracy or R².

- **Model Results**  
  Inspect predictions from the best model and download them as CSV.  
  This makes it easy to share outputs or continue analysis elsewhere.

---

## Code Walkthrough

### 1. Dataset Loading

We support three sources: sklearn datasets, pydataset, and uploaded CSVs.  
Caching (`@st.cache_data`) ensures data loads quickly without re-running on every refresh.

```python
@st.cache_data
def load_builtin_dataset(name: str):
    if name == "Iris":
        return datasets.load_iris(as_frame=True).frame
    elif name == "Wine":
        return datasets.load_wine(as_frame=True).frame
    elif name == "Breast Cancer":
        return datasets.load_breast_cancer(as_frame=True).frame
```


For pydataset, we normalize column names to avoid issues with dots or uppercase letters:

```python
@st.cache_data
def load_pydataset(name: str):
    df = pydata(str(name))
    df.columns = df.columns.str.replace('.', '_')
    df.columns = df.columns.str.lower()
    return df
```

---

### 2. EDA Tools
Quick stats and correlation heatmaps are generated with pandas and Plotly:

```python
def get_basic_stats(df):
    return df.describe(include="all").transpose()

def get_corr_heatmap(df):
    num_df = df.select_dtypes(include=["number"])
    corr = num_df.corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
    return fig
```

This gives users both tabular summaries and interactive visualizations.

---

### 3. Model Comparison (Light Mode)
The core of the app is a lightweight sklearn model leaderboard. We split the dataset, train multiple models, and score them.

```python
def run_sklearn_compare(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVC": SVC(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)
        results.append({"Model": name, "Score": score})

    return pd.DataFrame(results).sort_values("Score", ascending=False)
```

---

### 4. Model Results
Once the best model is selected, predictions are displayed and downloadable:
```python
df_pred = pd.DataFrame({"Actual": y_test, "Predicted": preds})
csv = df_pred.to_csv(index=False).encode("utf-8")

st.download_button("Download CSV", csv, "predictions.csv", "text/csv")
```

## Lessons Learned
- Streamlit multipage apps are a great way to organize workflows. Each page feels like a step in the pipeline
- Caching matters. Without it, even simple dataset loads can feel sluggish.
- Keep it light. Removing heavy dependencies makes deployment painless and avoids GPU package issues.
- Fallbacks are essential. Libraries evolve, and functions like pydataset.data() don’t always behave the same across versions.

---

## Next Steps
Some future additions:

- A search bar for pydataset datasets
- Time series, Multi modal datasets (Text, Image, Audio, Video)
- Feature importance plots for tree-based models
- Residual plots for regression tasks
- Possibly a model export option (pickle/ONNX) for downstream use

---

## Conclusion
This project reinforced how powerful Streamlit can be for rapid ML prototyping. By focusing on a lightweight design, I ended up with an app that’s easy to use, easy to deploy, and flexible enough for most everyday ML tasks.

You can explore the code and adapt it to your own workflows. Whether you’re teaching, experimenting, or just curious, this app provides a solid foundation for interactive machine learning in Python.

Enough talking, go and check the app - [App](https://end-to-end-ml-app.streamlit.app/)


##### References:
1. [Pydataset](https://pypi.org/project/pydataset/)