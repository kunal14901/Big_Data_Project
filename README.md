# Customer Churn Analysis 📊

A comprehensive machine learning project for predicting customer churn in subscription-based services using advanced data analysis and ensemble methods.

## 🎯 Project Overview

This project addresses the critical business problem of customer churn prediction for subscription-based services. By analyzing historical customer data, we developed machine learning models to identify customers likely to cancel their subscriptions, enabling proactive retention strategies.

### Key Achievements
- **94% Accuracy** using Random Forest Classifier with SMOTEENN
- Identified critical churn factors including contract type, tenure, and service features
- Developed actionable insights for customer retention strategies

## 👥 Team Members

- **Aadhar Mehta** (20MA20001)
- **Ajitesh Bharti** (20MA20004)
- **Chitresh Choudhary** (20MA20019)
- **Esha Jain** (20MA20021)
- **Kunal Kumar** (20MA20028)
- **Lovely Kaur** (20MA20031)
- **Pooja Sonje** (20MA20055)
- **Princy** (20MA20046)
- **Sonal** (20MA20054)
- **Yash Laxkar** (20MA20066)

**Supervisor:** Professor Bibhas Adhikari, Department of Mathematics, IIT Kharagpur

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Models & Results](#models--results)
- [Installation & Usage](#installation--usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [References](#references)

## 🎯 Problem Statement

### Background
A subscription-based service company experiencing high churn rates where customers cancel before subscription periods end, resulting in revenue loss and hindered growth.

### Objective
Develop a churn prediction model to accurately identify at-risk customers, enabling proactive retention measures such as:
- Targeted promotions
- Service quality improvements
- Personalized offers

### Key Challenges
- **Imbalanced Dataset**: 27% churned vs 73% non-churned customers
- **Feature Engineering**: Identifying relevant predictive features
- **Dynamic Churn Patterns**: Adapting to changing customer behavior
- **Real-time Prediction**: Low-latency prediction requirements

## 📊 Dataset

- **Size**: Thousands of customer records
- **Features**: Demographics, subscription details, usage behavior, service features
- **Target Variable**: Churn status (binary: churned/not churned)
- **Imbalance Ratio**: 27% churned, 73% retained

### Data Preprocessing
- Removed irrelevant features (customer ID, columns with >99% missing values)
- Handled missing values strategically
- Converted categorical variables using one-hot encoding
- Applied tenure grouping for dimensionality reduction

## 🔍 Methodology

### 1. Data Cleaning & Preparation
- Comprehensive data quality assessment
- Missing value analysis and treatment
- Feature selection and engineering
- Data type conversions and encoding

### 2. Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Individual feature distributions
- **Bivariate Analysis**: Feature relationships with churn
- **Correlation Analysis**: Feature interdependencies
- **Visualization**: Count plots, KDE plots, heatmaps

### 3. Model Development
- **Baseline Models**: Decision Tree, Random Forest
- **Imbalance Handling**: SMOTEENN (SMOTE + Edited Nearest Neighbors)
- **Dimensionality Reduction**: Principal Component Analysis (PCA)
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score

## 🔑 Key Findings

### High Churn Indicators
- **Contract Type**: Month-to-month contracts (highest risk)
- **Payment Method**: Electronic check payments
- **Service Features**: No online security, no tech support
- **Internet Service**: Fiber optic customers
- **Tenure**: First-year customers (1-12 months)
- **Charges**: High monthly charges with low total charges

### Low Churn Indicators
- **Contract Type**: Long-term contracts (1-2 years)
- **Service**: Customers without internet service
- **Tenure**: Long-term customers (5+ years)
- **Demographics**: Customers with dependents

### Neutral Factors
- Gender
- Phone service availability
- Multiple lines
- Streaming services (TV/Movies)

## 🤖 Models & Results

### 1. Decision Tree Classifier
- **Base Model**: 80% accuracy, F1-score: 0.55 (churned class)
- **With SMOTEENN**: 94% accuracy, improved minority class performance

### 2. Random Forest Classifier ⭐ **Best Model**
- **Base Model**: 79% accuracy, F1-score: 0.52 (churned class)
- **With SMOTEENN**: 94% accuracy, balanced precision-recall

### 3. Principal Component Analysis
- **Result**: 74% accuracy, F1-score: 0.69
- **Conclusion**: Dimensionality reduction decreased performance

### Final Model Selection
**Random Forest Classifier with SMOTEENN** achieved the best performance:
- **Accuracy**: 94%
- **Balanced performance** across both classes
- **Robust predictions** for minority class (churned customers)

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Basic Usage
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN

# Load your data
data = pd.read_csv('customer_data.csv')

# Preprocess data (refer to our preprocessing pipeline)
X_processed, y = preprocess_data(data)

# Apply SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteenn.fit_resample(X_processed, y)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)

# Make predictions
predictions = rf_model.predict(X_test)
```

## 🛠️ Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms
- **Imbalanced-learn**: Handling imbalanced datasets
- **SMOTE**: Synthetic data generation
- **Jupyter Notebook**: Development environment

## 📈 Business Impact

### Actionable Insights
1. **Target Month-to-Month Customers**: Highest churn risk group
2. **Improve Service Add-ons**: Focus on online security and tech support
3. **Optimize Pricing Strategy**: Address high monthly charges for new customers
4. **Payment Method Strategy**: Reduce reliance on electronic check payments
5. **Early Intervention**: Implement retention programs for first-year customers

### ROI Potential
- **Proactive Retention**: Reduce churn rate through targeted interventions
- **Revenue Protection**: Prevent subscription cancellations
- **Cost Efficiency**: Focus retention efforts on high-risk customers


## 📝 License

This project is part of an academic assignment at IIT Kharagpur. Please respect academic integrity guidelines when using this code.

## 🔗 References

1. [End-to-End ML Project - Telco Customer Churn](https://towardsdatascience.com/end-to-end-machine-learning-project-telco-customer-churn-90744a8df97d)
2. [Predicting Customer Churn](https://medium.com/@islamhasabo/predicting-customer-churn-bc76f7760377)
3. [Random Forest Algorithm - JavaTpoint](https://www.javatpoint.com/machine-learning-random-forest-algorithm)
4. [SMOTE for Imbalanced Classification](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)
5. [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

⭐ **Star this repository** if you found it helpful!

📧 **Contact**: For questions or collaborations, reach out to any of the team members listed above.

---
*This project was completed as part of the Big Data Analysis course at IIT Kharagpur (Jan 2023 - Apr 2023)*
