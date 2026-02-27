# AI Tools and Development Environment

## Overview
This document outlines all tools, libraries, frameworks, and development environments used to build the **Mercato Uplift Analytics Dashboard** for the DataVerse Datathon.

---

## 🛠️ Development Environment

### Primary Development Platform
- **Google Colab** (Primary ML environment)
  - Purpose: Fast prototyping, GPU availability, easy data visualization
  - Python Version: 3.10+
  - Used for model training and experimentation

### Alternative Environments
- **Jupyter Notebook**
  - Local development and data exploration
  - Feature engineering and statistical analysis

- **VS Code**
  - Dashboard HTML/CSS/JavaScript development
  - Code editing and version control

---

## 📊 Data Analysis Stack

### Core Libraries
| Library | Version | Purpose |
|---------|---------|----------|
| **Python** | 3.10+ | Primary programming language |
| **Pandas** | 2.0+ | Data manipulation and cleaning |
| **NumPy** | 1.24+ | Numerical computing and array operations |
| **Matplotlib** | 3.7+ | Statistical visualization |
| **Seaborn** | 0.12+ | Advanced data visualization |

### Key Usage
- Dataset cleaning and preprocessing
- Feature engineering (promo_dependency, purchase_frequency)
- Statistical analysis and correlation studies
- Exploratory Data Analysis (EDA)

---

## 🤖 Machine Learning Stack

### Uplift Modeling (T-Learner)
**Framework:** scikit-learn

**Algorithm:** Random Forest Classifier
- **Model 1 (Treatment):** Trained on promoted customers
- **Model 2 (Control):** Trained on non-promoted customers
- **CATE Calculation:** μ₁(x) - μ₀(x)

**Purpose:**
- Estimate incremental effect of promotions
- Calculate Conditional Average Treatment Effect (CATE)
- Segment customers into:
  - Sure Things
  - Persuadables
  - Lost Causes
  - Sleeping Dogs

### Churn Prediction
**Framework:** XGBoost

**Algorithm:** XGBoost Classifier
- **Target:** churn = no purchase within 90 days
- **ROC-AUC Score:** ~0.89

**Key Features:**
- promo_dependency
- purchase_frequency
- customer_support_calls
- membership_duration
- transaction_history

**Interpretability:**
- **SHAP (SHapley Additive exPlanations)**
  - Feature importance visualization
  - Top churn driver: promo_dependency

### Additional ML Tools
| Tool | Purpose |
|------|----------|
| **scikit-learn** | Model training, evaluation, metrics |
| **XGBoost** | Gradient boosting for churn prediction |
| **SHAP** | Model interpretability and feature importance |
| **imbalanced-learn** | Handling class imbalance (if needed) |

---

## 🎨 Dashboard Development

### Frontend Technologies
| Technology | Version | Purpose |
|------------|---------|----------|
| **HTML5** | - | Page structure and content |
| **CSS3** | - | Styling, animations, responsive design |
| **JavaScript (Vanilla)** | ES6+ | Interactive logic and simulations |
| **Chart.js** | 4.4.1 | Data visualization library |

### Visualization Components
**Chart.js 4.4.1** is used for:
- 📊 Bar charts (revenue comparison)
- 📈 Uplift distribution charts
- 🔴 Churn visualization
- 📉 ROC curve plotting
- 📊 Lift chart
- 🎯 SHAP feature importance bars

**Custom JavaScript Features:**
- Interactive budget allocation simulator
- Real-time P&L calculations
- Dynamic strategy rating system
- Responsive UI updates

---

## 🧠 Model Training Pipeline

### Step 1: Data Preparation
```python
# Feature Engineering
features = [
    'promo_dependency',
    'purchase_frequency', 
    'membership_duration',
    'support_calls',
    'category_preference'
]

# Target variables
promotion_flag  # Treatment indicator
purchase_outcome  # Conversion target
```

### Step 2: Train-Test Split
- **80%** Training data
- **20%** Testing data
- **Purpose:** Prevent overfitting, evaluate generalization

### Step 3: T-Learner Training
```python
# Treatment Model (μ₁)
model_treatment = RandomForestClassifier()
model_treatment.fit(X_treatment, y_treatment)

# Control Model (μ₀)
model_control = RandomForestClassifier()
model_control.fit(X_control, y_control)

# CATE Calculation
CATE = model_treatment.predict_proba(X)[:, 1] - model_control.predict_proba(X)[:, 1]
```

### Step 4: Customer Segmentation
| Segment | Logic | Business Action |
|---------|-------|------------------|
| **Sure Things** | High μ₀, High μ₁ | Convert without promotion |
| **Persuadables** | Low μ₀, High μ₁ | Target with promotions |
| **Lost Causes** | Low μ₀, Low μ₁ | Do not promote |
| **Sleeping Dogs** | High μ₀, Low μ₁ | Avoid promotion |

### Step 5: Churn Model Training
```python
# XGBoost Churn Predictor
import xgboost as xgb

churn_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6
)

churn_model.fit(X_train, y_churn)

# Model Evaluation
roc_auc = roc_auc_score(y_test, churn_model.predict_proba(X_test)[:, 1])
# ROC-AUC ≈ 0.89
```

---

## 📁 Project Structure

```
mercato-uplift-dashboard/
│
├── data/
│   ├── transactions.csv
│   └── cleaned_dataset.csv
│
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── uplift_model_training.ipynb
│   └── churn_model_training.ipynb
│
├── models/
│   ├── treatment_model.pkl
│   ├── control_model.pkl
│   └── churn_xgboost.pkl
│
├── dashboard/
│   └── entropy_dashboard.html
│
├── scripts/
│   ├── feature_engineering.py
│   ├── uplift_training.py
│   └── churn_training.py
│
└── README.md
```

---

## 🎯 Dashboard System Architecture

### Three-Layer Analytics

**Layer 1: Descriptive Analytics**
- Current revenue metrics
- Margin loss analysis
- Churn rate comparison

**Layer 2: ML Insights**
- Uplift segmentation (CATE distribution)
- Churn prediction probabilities
- Feature importance (SHAP values)

**Layer 3: Prescriptive Decision Engine**
- Interactive budget allocation simulator
- Real-time ROI calculations
- Strategy rating system

### Financial Outputs
- 💰 Projected revenue
- 📊 Margin recovered
- 📈 Annual P&L change
- ⭐ Strategy effectiveness rating

---

## 💡 Technical Explanation (20-Second Pitch)

**For Judges:**

> "We trained a **T-Learner uplift model** using Random Forest to estimate the incremental effect of promotions, combined it with an **XGBoost churn predictor**, and built an interactive prescriptive dashboard using **HTML, JavaScript, and Chart.js** to simulate optimal promotion allocation."

**Key Technical Highlights:**
- ✅ **Causal ML** (not just predictive modeling)
- ✅ **T-Learner architecture** (treatment vs control)
- ✅ **CATE-based segmentation** (individual-level effects)
- ✅ **Interpretable AI** (SHAP values)
- ✅ **Prescriptive analytics** (actionable simulator)

---

## 🔧 Installation & Setup

### Python Dependencies
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap
```

### Google Colab Setup
```python
# No installation needed - pre-installed libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap
```

### Dashboard Setup
- No server required
- Open `entropy_dashboard.html` in any modern browser
- Chart.js loaded via CDN

---

## 📝 Model Evaluation Metrics

### Uplift Model
- **Qini Coefficient:** Measures uplift quality
- **Lift Chart:** Visual uplift validation
- **Treatment Effect Distribution:** CATE histogram

### Churn Model
- **ROC-AUC:** ~0.89
- **Precision-Recall:** Balanced for business use
- **Feature Importance:** SHAP-based ranking

---

## 🚀 Why This Stack?

| Choice | Reason |
|--------|--------|
| **Random Forest for Uplift** | Handles non-linear treatment effects |
| **XGBoost for Churn** | Best-in-class gradient boosting |
| **T-Learner Architecture** | Simple, interpretable causal inference |
| **Chart.js** | Lightweight, fast, interactive |
| **Google Colab** | Free GPU, collaborative, reproducible |
| **Vanilla JS** | No framework overhead, fast load |

---

## 📞 Technical Contact

**Repository:** [GitHub - DataVerse](https://github.com/Harnitya29/DataVerse)
**Team:** Team Entropy
**Event:** DataVerse Datathon 2026

---

**Last Updated:** February 27, 2026
