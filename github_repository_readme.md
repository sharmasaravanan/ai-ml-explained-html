# AI & ML Interactive Course

A self-contained, interactive learning repository covering the foundations of Artificial Intelligence, Classical Machine Learning, and Deep Learning. Each module pairs clear visual explanations with browser-based interactive demos, quizzes, and hands-on Python examples.

---

## Repository Structure

| File | Part / Module | Description |
| :--- | :--- | :--- |
| `01-ai-ml-dl.html` | **Part 1 · Module 1** | **What is AI, ML, and Deep Learning?** — Interactive nested diagrams, AI history & winters timeline, learning paradigms game, and a rule-based vs. learned spam filter demo. |
| `02-ml-workflow.html` | **Part 1 · Module 2** | **The ML Workflow** — Step-by-step pipeline explorer, customer churn case study, train/validation/test splits, interactive data leakage simulator, and debugging scenarios. |
| `03-data-preprocessing.html` | **Part 1 · Module 3** | **Data and Preprocessing** — Interactive labs for cleaning messy text, handling missing values/outliers, categorical encoding, feature scaling, and stratified train/test splitting. |
| `part2-classical-ml.html` | **Part 2 · Topics 4–8** | **Classical Machine Learning** — Covers Linear/Logistic Regression, Decision Trees & Random Forests, SVM/KNN/Naive Bayes, Model Evaluation metrics, and Unsupervised Learning (K-Means & PCA). |
| `part3-deep-learning.html` | **Part 3 · Topics 9–12** | **Deep Learning** — Covers Neural Networks from scratch, CNNs & Computer Vision, RNNs & NLP basics, and Transformers, featuring live canvas demos for gradient descent, convolution filters, and self-attention. |
| `data_preprocessing.py` | **Hands-On Lab** | A leak-proof `scikit-learn` preprocessing and modeling script using `ColumnTransformer` and `Pipeline` (accompanies Part 1, Module 3). |
| `customer_churn_preprocessing.csv` | **Dataset** | Sample customer churn dataset containing messy city strings, missing values, and outliers used by `data_preprocessing.py`. |

---

## Getting Started

### 1. Viewing the Interactive Course Modules
All `.html` modules are completely self-contained with zero external dependencies or build steps required.

1. Clone or download this repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Open `01-ai-ml-dl.html` directly in any modern web browser and use the navigation links at the bottom of each page to progress through the course.

> **Tip:** You can also enable **GitHub Pages** in your repository settings (`Settings → Pages → Deploy from branch`) to host and share the interactive modules directly on the web.

### 2. Running the Python Hands-On Script
To run the preprocessing pipeline script locally, ensure you have Python 3.8+ installed along with `numpy`, `pandas`, and `scikit-learn`.

```bash
pip install numpy pandas scikit-learn
python data_preprocessing.py
```

---

## Key Concepts Covered

- **Foundations:** AI vs. ML vs. DL vs. Generative AI, supervised/unsupervised/reinforcement learning, end-to-end ML pipelines, and preventing data leakage.
- **Data Engineering:** Text normalization, median/mean imputation, one-hot & ordinal encoding, Min-Max & Z-score standardization, and stratified splitting.
- **Classical Algorithms:** Gradient descent, L1/L2 regularization, Gini/Entropy splits, bagging, margin maximization, confusion matrices, ROC-AUC, cross-validation, K-Means, and PCA.
- **Deep Learning:** Activation functions, backpropagation, convolutional kernels, transfer learning, sequence modeling (LSTM/GRU), and query-key-value self-attention.