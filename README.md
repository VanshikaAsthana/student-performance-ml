# Early Risk Detection for Student Academic Performance
## 📌 Problem Statement

Academic failure is often identified too late, after final exams.
This project focuses on early detection of at-risk students using machine learning, enabling timely academic intervention.

Instead of predicting raw marks, the system classifies students into risk categories (Low / Medium / High) based on academic and behavioral indicators.

## 📊 Dataset

The dataset contains student-level information including:

Study habits

Attendance

Academic background

Previous assessment scores

Only pre-final / mid-semester features are considered to avoid data leakage.

## 🎯 Objective

Identify students at risk of poor performance early

Build an interpretable ML model

Analyze key factors contributing to academic risk

## 🧠 Approach

Exploratory Data Analysis (EDA)

Feature engineering

Risk category creation

Model training & evaluation

Explainability & bias analysis

## 📈 Models Used

Logistic Regression (baseline)

Random Forest

Gradient Boosting

Evaluation metrics prioritize Recall and F1-score to minimize false negatives (missing at-risk students).

## 🔍 Key Insights

Attendance and prior scores are the strongest predictors

Performance trends matter more than absolute scores

Models can show bias if not evaluated across subgroups

## 🤖 Modeling Summary

Converted final grade into pass/fail classification

Trained Logistic Regression with class balancing

Evaluated using recall and F1-score

Interpreted predictions using model coefficients

## ⚠️ Ethical Considerations

Predictions should assist educators, not penalize students

Model decisions must be interpretable

Bias across demographic groups is explicitly evaluated

## 🚀 Future Work

Real-time dashboard for educators

Integration with institutional systems

Personalized intervention recommendations
