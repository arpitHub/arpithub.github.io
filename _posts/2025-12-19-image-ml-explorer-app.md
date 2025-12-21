---
layout: post
title: "Image Explorer App: End-to-End ML for Image Classification"
subtitle: "Learn how to preprocess, model, and interpret image data interactively"
cover-img: /assets/img/image-explorer-cover.png
thumbnail-img: /assets/img/image-explorer-main.png
share-img: /assets/img/image-explorer-cover.png
gh-repo: arpithub/arpithub.github.io
gh-badge: [star, fork, follow]
tags: [datascience,ml,image-classification,computer-vision,streamlit,pipeline]
comments: true
---

Text data surrounds us — from emails and reviews to tweets and articles. The **Text ML Explorer App** was built to help learners understand how machine learning models can classify text step by step. It’s an interactive Streamlit app that takes you from raw text to predictions, making the workflow transparent and approachable.

## Motivation

Computer vision often feels like a black box to learners. Images are transformed into arrays, features are extracted, and models make predictions — but the steps in between are rarely visible. The **Image ML Explorer App** was designed to make these steps **transparent, interactive, and educational**, so students can see how raw pixels become predictions.

---

## Features

The app is organized into clear pages, each representing a stage in the image ML pipeline:

- **Dataset Explorer 🖼️**  
  Preview sample image datasets (e.g., CIFAR‑10, MNIST) or upload your own.  
  Inspect image dimensions, channels, and labels.

- **Preprocessing 🔧**  
  - Convert images to grayscale or resize them.  
  - Normalize pixel values.  
  - Visualize transformations side‑by‑side.  

- **Model Builder 🤖**  
  - Train simple CNNs or transfer learning models (e.g., ResNet, VGG).  
  - Compare accuracy across architectures.  
  - Visualize training curves (loss/accuracy).  

- **Results 📊**  
  - Test custom images against trained models.  
  - Display predictions with confidence scores.  
  - Show Grad‑CAM or saliency maps to highlight what the model “sees.”  

---

## Code Walkthrough
### 1. 📂 Image Dataset Explorer

This page introduces how image datasets are structured, loads the default **Digits dataset** from scikit‑learn, and lets students preview sample images:

```python
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np

st.title("📂 Image Dataset Explorer")

# --- Explanations ---
st.write("### How Image Datasets Work")
st.markdown("""
Images are stored as arrays of pixel values.  
For example, a grayscale image is a 2D matrix where each number represents brightness (0 = black, 255 = white).

👉 Machine learning models learn patterns in these numbers to classify images.
""")

# --- Load default dataset (Digits) ---
digits = load_digits()

X = digits.images   # shape (n_samples, 8, 8)
y = digits.target   # labels (0–9)

st.write("Dataset shape:", X.shape)
st.write("Number of classes:", len(np.unique(y)))

# --- Show sample images ---
st.write("### Sample Images")
num_samples = st.slider("Select number of samples to view:", 4, 20, 8)

fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
for i, ax in enumerate(axes):
    ax.imshow(X[i], cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")
st.pyplot(fig)

# --- Save for later pages ---
st.session_state["X_images"] = X
st.session_state["y_labels"] = y
```

👉 **Teaching note:**  
The default dataset here is Digits, a classic computer vision dataset of handwritten numbers (0–9). It’s widely used in tutorials because:

 - Each image is small (8×8 pixels), making it computationally lightweight.
 - Labels are intuitive (digits 0–9), so learners can quickly connect predictions to real outcomes.
 - It provides a simple entry point into image classification before moving to larger datasets like MNIST or CIFAR‑10.


 ### 2. 🔧 Preprocessing

This page demonstrates how raw images are transformed into usable features for machine learning. Students can experiment with resizing, grayscale conversion, and normalization.

```python
import streamlit as st
import numpy as np
from PIL import Image

st.title("🔧 Image Preprocessing")

# --- Explanations ---
st.write("### Why Preprocessing Matters")
st.markdown("""
Raw images are large arrays of pixel values.  
Preprocessing reduces complexity and highlights important features.  
👉 Without preprocessing, models may struggle to learn efficiently.
""")

# --- Load image from session state ---
if "X_images" in st.session_state:
    img = st.session_state["X_images"][0]  # take first sample for demo
    st.image(img, caption="Original Image")

    # Resize
    resized = Image.fromarray(img).resize((16, 16))
    st.image(resized, caption="Resized (16x16)")

    # Grayscale
    gray = Image.fromarray(img).convert("L")
    st.image(gray, caption="Grayscale")

    # Normalize
    arr = np.array(img) / 255.0
    st.write("Normalized pixel values (first 5x5 block):")
    st.write(arr[:5, :5])
```
👉 **Teaching note:**  
Preprocessing is the bridge between raw data and machine learning models. By visualizing each step:

 - Resizing reduces dimensionality, making training faster.
 - Grayscale simplifies color images into intensity values.
 - Normalization scales pixel values to [0,1], improving model stability.

This page helps learners see how preprocessing transforms images from raw pixels into structured inputs for models.

### 3. 🤖 Model Builder

This page introduces how to train a simple classifier on image data. It uses scikit‑learn’s `LogisticRegression` to classify the digits dataset.

```python
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("🤖 Model Builder")

# --- Explanations ---
st.write("### Training a Simple Model")
st.markdown("""
We’ll use **Logistic Regression** to classify digits (0–9).  
👉 Logistic Regression is a simple yet effective baseline for image classification.
""")

# --- Load data from session state ---
if "X_images" in st.session_state and "y_labels" in st.session_state:
    X = st.session_state["X_images"]
    y = st.session_state["y_labels"]

    # Flatten images (8x8 → 64 features)
    X_flat = X.reshape((X.shape[0], -1))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"Model trained! Accuracy: {acc:.2f}")

    # Save model for later pages
    st.session_state["model"] = model
```

👉 **Teaching note:**  
This page shows learners how to move from preprocessed data to a trained model. Key insights:

 - Images must be flattened into feature vectors before training.
 - Logistic Regression, though simple, provides a strong baseline for classification tasks.
 - Accuracy gives a quick measure of performance, but later pages can explore deeper evaluation (confusion matrices, misclassified samples).

By starting with Logistic Regression, students learn the fundamentals of model training before experimenting with more complex algorithms like CNNs.

### 4. 📊 Results

This page lets learners test the trained model on new samples and visualize predictions.

```python
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

st.title("📊 Results")

# --- Explanations ---
st.write("### Testing the Model")
st.markdown("""
Now we can evaluate the trained model on unseen data.  
👉 This step shows how well the model generalizes beyond the training set.
""")

# --- Load model and data ---
if "model" in st.session_state and "X_images" in st.session_state and "y_labels" in st.session_state:
    model = st.session_state["model"]
    X = st.session_state["X_images"]
    y = st.session_state["y_labels"]

    # Flatten images
    X_flat = X.reshape((X.shape[0], -1))

    # Predict
    y_pred = model.predict(X_flat)

    # Show confusion matrix
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)

    # Try a single sample
    idx = st.slider("Select a sample index:", 0, len(X) - 1, 0)
    st.image(X[idx], caption=f"True Label: {y[idx]}")
    st.write(f"Predicted Label: {y_pred[idx]}")
    ```

👉 **Teaching note:**  
The Results page is where theory meets practice. Students can:

 - See how well the model performs across all classes using a confusion matrix.
 - Select individual samples to compare true vs predicted labels.
 - Understand that even simple models like Logistic Regression can make mistakes — and those mistakes are valuable learning opportunities.

This final step closes the loop: from dataset → preprocessing → model building → evaluation.