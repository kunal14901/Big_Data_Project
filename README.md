# Customer Churn Analysis 📊

Machine learning project to predict customer churn in subscription-based services with **94% accuracy**.

## 🎯 Overview

Developed predictive models to identify customers likely to cancel subscriptions, enabling proactive retention strategies for a subscription service company.

**Key Results:**
- 94% accuracy using Random Forest + SMOTEENN
- Identified critical churn factors (contract type, tenure, service features)
- Addressed imbalanced dataset (27% churn rate)

## 👥 Team
**IIT Kharagpur** - Big Data Analysis Project (Jan-Apr 2023)

Aadhar Mehta, Ajitesh Bharti, Chitresh Choudhary, Esha Jain, Kunal Kumar, Lovely Kaur, Pooja Sonje, Princy, Sonal, Yash Laxkar

**Supervisor:** Prof. Bibhas Adhikari

## 🔍 Key Findings

### High Churn Risk
- Month-to-month contracts
- Electronic check payments
- No online security/tech support
- Fiber optic internet users
- First-year customers (1-12 months)

### Low Churn Risk
- Long-term contracts (1-2 years)
- Customers with dependents
- 5+ years tenure

## 🤖 Models & Performance

| Model | Accuracy | F1-Score (Churn) |
|-------|----------|------------------|
| Decision Tree | 80% | 0.55 |
| **Random Forest + SMOTEENN** | **94%** | **High** |
| PCA | 74% | 0.69 |

## 🚀 Quick Start

```python
# Install dependencies
pip install pandas scikit-learn imbalanced-learn seaborn

# Load and preprocess data
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN

# Train model
smoteenn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteenn.fit_resample(X, y)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)
```

## 🛠️ Tech Stack
- **Python**: Pandas, NumPy, Scikit-learn
- **ML**: Random Forest, SMOTEENN, PCA
- **Visualization**: Matplotlib, Seaborn

## 📈 Business Impact
- **Proactive retention** for high-risk customers
- **Revenue protection** through churn prevention
- **Targeted strategies** for month-to-month subscribers

---
⭐ Star this repo if helpful! | 📧 Contact team for questions
