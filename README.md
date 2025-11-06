<div align="center">

# 🎯 Feature Extraction, Feature Selection & Optimization Analysis

**Notebook:** `CDM__2_Final.ipynb`  
**Course:** Computational Data Mining  
**Author:** AmirHossein Talebi Koohestani  
**University:** Amir Kabir University of Technology (Tehran Polytechnic)  
**Supervisor:** Dr. Mehdi Ghatee  
**TA:** Dr. Behnam Yousefimehr
</div>

---

## 📖 Project Overview

This notebook focuses on analyzing how **feature extraction**, **feature selection**, and **optimization strategies** affect model performance, stability, and interpretability across classification, regression, and clustering tasks.

We explore:

- Linear dependency and redundancy in feature spaces  
- Dimensionality reduction using **Principal Component Analysis (PCA)**  
- Feature filtering approaches (e.g., **SelectKBest**)  
- Wrapper-based selection via **Recursive Feature Elimination (RFE)**  
- Sensitivity of different optimization methods in model training  

The experiments highlight how thoughtful feature design can improve performance, prevent overfitting, and accelerate convergence.

---

## 📦 Datasets Used

| Task | Dataset | Source |
|------|---------|--------|
| **Classification** | Breast Cancer Dataset | `sklearn.datasets.load_breast_cancer()` |
| **Regression** | Boston Housing Dataset | https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset |
| **Clustering** | Iris Dataset | `sklearn.datasets.load_iris()` |

---

## 🚀 Installation & Setup

### Requirements

- Python **3.8+**
- Jupyter Notebook / JupyterLab

### Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

Then launch the notebook:

```bash
jupyter notebook
```

Open and run:

```
CDM__2_Final.ipynb
```

---

## 🎯 Objectives

- Examine feature collinearity and understand where redundant information exists.
- Reduce dimensionality using PCA and evaluate variance retention.
- Compare selection performance of SelectKBest (filter) vs. RFE (wrapper).
- Evaluate classification/regression performance before and after feature selection.
- Analyze clustering robustness using reduced feature spaces.
- Compare gradient-based vs. iterative/heuristic optimization strategies.

---

## 🧠 Techniques Implemented

| Method | Description |
|--------|-------------|
| **PCA (Principal Component Analysis)** | Extracts orthogonal components maximizing variance, reducing dimensionality while preserving structure. |
| **SelectKBest** | Ranks features using statistical scoring functions; fast and model-agnostic. |
| **RFE (Recursive Feature Elimination)** | Iterative model-based feature pruning using trained estimators. |
| **Optimization Strategies** | Comparison of solvers (e.g., SGD) and convergence behavior. |

---

## 🔬 Methodology

1. Load and preprocess each dataset.
2. Analyze baseline model performance.
3. Apply PCA for dimensionality reduction & examine explained variance.
4. Perform feature selection (SelectKBest & RFE) and compare selected subsets.
5. Train and evaluate models using:
   - Original features
   - PCA-transformed features
   - Selected features
6. Compare optimizer performance on the same model architecture.
7. Analyze accuracy, MSE, silhouette score, and convergence metrics.

---

## 📊 Evaluation Metrics

| Task | Metrics |
|------|---------|
| Classification | Accuracy |
| Regression | MSE |
| Clustering | Inertia, Silhouette Score |
| PCA Quality | Explained Variance Ratio |
| Optimizer Comparison | Convergence speed, Stability, Final loss |

---

## 📝 Key Findings (General Insights)

- PCA can significantly reduce dimensionality **while retaining most variance**.
- SelectKBest is faster but may overlook feature interactions.
- RFE yields more refined subsets but is computationally expensive.
- Reduced and selected feature spaces often **improve generalization**.
---

## 📚 References

- Scikit-Learn Documentation — https://scikit-learn.org/stable/
- Kaggle: Boston Housing Dataset - https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset
