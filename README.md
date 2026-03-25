# 🚀 Holiday Package Purchase Prediction (Random Forest Model)

## 📌 Overview

This project focuses on predicting whether a customer will purchase a holiday package (`ProdTaken`) using machine learning.

After experimenting with multiple models, **Random Forest** provided the best performance. Therefore, further improvement was done using **hyperparameter tuning**, and the final results are based on the tuned model.

---

## 🎯 Problem Statement

Build a classification model to predict customer purchase behavior based on features like:

* Age
* Monthly Income
* Duration of Pitch
* Number of Trips
* Type of Contact
* And more

---

## 🛠️ Tech Stack

* Python 🐍
* Pandas, NumPy
* Scikit-learn
* Matplotlib / Seaborn

---

## ⚙️ Workflow

### 1️⃣ Data Preprocessing

* Handled missing values:

  * Median → Numerical features
  * Mode → Categorical features
* Feature selection
* Train-test split

---

### 2️⃣ Model Selection

* Trained multiple models:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Gradient Boosting

👉 **Random Forest performed the best**, so it was selected for further tuning.

---

### 3️⃣ Hyperparameter Tuning

Used **RandomizedSearchCV** to optimize Random Forest:

* Tuned parameters:

  * `n_estimators`
  * `max_depth`
  * `min_samples_split`
  * `min_samples_leaf`
  * `max_features`

---

## 📊 Final Results (After Hyperparameter Tuning)

### 🔹 Random Forest Classifier

**Training Set Performance**

* Accuracy: **1.0000**
* F1 Score: **1.0000**
* Precision: **1.0000**
* Recall: **1.0000**
* ROC AUC Score: **1.0000**

**Test Set Performance**

* Accuracy: **0.9305**
* F1 Score: **0.9253**
* Precision: **0.9695**
* Recall: **0.6649**
* ROC AUC Score: **0.8299**

---

## 🔍 Key Insights

* Hyperparameter tuning significantly improved overall performance
* High training accuracy indicates **overfitting**
* Model achieves **high precision**, meaning predictions are very reliable
* Recall is comparatively lower, indicating some positive cases are missed

---

## ⚠️ Challenges

* Overfitting due to highly complex Random Forest
* Trade-off between precision and recall
* Handling class imbalance

---

## 🚀 Future Improvements

* Apply **class balancing techniques** (SMOTE / class_weight)
* Tune decision threshold to improve recall
* Try advanced models like XGBoost / LightGBM
* Perform feature engineering for better generalization

---

## ▶️ How to Run

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
jupyter notebook
```

---

## 📬 Contact

Feel free to reach out for any questions or collaboration!

---
