# 🏦 Loan Approval Prediction System

A Machine Learning web application that predicts whether a loan should be:

- ✅ Approved (1)
- ❌ Rejected (0)

Built using real-world financial data and deployed using Flask.

---

## 📌 Problem Statement

Financial institutions need to automate loan approval decisions to:

- Reduce default risk
- Improve decision-making speed
- Maintain consistent credit policies

This project predicts loan approval using applicant financial data.

---

## 📊 Dataset Overview

- Total Records: **45,000**
- Target Variable:
  - `loan_status` → 1 (Approved), 0 (Rejected)

### ⚠️ Data Imbalance:
- Rejected (0): 35,000  
- Approved (1): 10,000  

👉 The dataset is **imbalanced**, which affects model behavior.

---

## 🧠 Model Used

### ✅ Final Model: Random Forest Classifier

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42
)