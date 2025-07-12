# Financial Fraud Detection Using Decision Tree Models

**Author:** Paul Carmody  
**MSc Computer Science with Data Analytics, Northumbria University (2024/25)**

---

## Overview

This project explores the application and evolution of decision tree models and advanced ensemble machine learning methods—such as Random Forests, Gradient Boosting, and Graph Neural Networks—in financial fraud detection. The project is based on a comprehensive MSc report examining both technical and ethical aspects of machine learning for large-scale fraud prevention, with reference to real industry use cases.

---

## Objectives

- Investigate how decision tree models and ensembles are used for fraud detection in finance.
- Understand the strengths and limitations of different approaches.
- Review real-world case studies (e.g., American Express, HSBC).
- Discuss emerging methods (XGBoost, LightGBM, GNNs) and explainable AI.
- Reflect on ethical issues including bias, transparency, and compliance.

---

## Key Insights

- **Decision Trees** offer simple, interpretable rules, but can overfit on complex data.
- **Random Forests** aggregate many trees to improve accuracy and robustness, widely used in banking for real-time fraud detection.
- **Gradient Boosting Machines** (XGBoost, LightGBM) deliver state-of-the-art performance, uncovering subtle fraud patterns.
- **Graph Neural Networks** are emerging for relational data (e.g., finding fraud rings in transaction networks).
- **Explainable AI (XAI):** Tools like SHAP and LIME are essential for regulatory transparency and trust in model predictions.
- **Ethics:** Highlights risks such as model bias and the importance of human oversight.

---

## Industry Case Studies

- **American Express:**  
  Uses a random forest model (1,000+ trees) for real-time analysis of over $1 trillion in annual transactions, reducing fraud and false positives.

- **HSBC:**  
  Combines decision trees with advanced ML and Google Cloud AI to identify 2-4x more suspicious activities than traditional systems, while cutting false positives by 60%.

---

## Technical Concepts Illustrated

```python
# Example: Decision Tree Pseudocode
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
