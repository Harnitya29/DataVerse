# Mercato Uplift Analytics Dashboard
## Team Entropy - DataVerse Datathon 2026

🎯 **Interactive Machine Learning Dashboard for Optimizing Promotional Spend using Causal Inference**

---

## 📋 Problem Statement

Retail companies waste **millions annually** on misallocated promotional budgets:
- 💰 Spending heavily on **"Sure Things"** (customers who'll buy anyway)
- 💸 Targeting **"Lost Causes"** (who won't respond to incentives)
- 📉 Missing **"Persuadables"** (customers with highest uplift from promotions)

**The Challenge:** Standard predictive models identify *likely buyers*, not *incremental buyers*.

---

## 🎯 Objective

Build an **interactive prescriptive analytics dashboard** that:
1. ✅ Identifies customer segments based on **causal treatment effects** (not correlation)
2. ✅ Predicts **churn risk** using interpretable ML (SHAP values)
3. ✅ Simulates **ROI impact** of different promotional strategies
4. ✅ Provides **actionable insights** for budget allocation
5. ✅ Demonstrates **advanced causal ML** using T-Learner architecture

---

## 📊 Solution: Entropy-Final.html Dashboard

### Three-Layer Analytics System

#### Layer 1: Descriptive Analytics
- Current revenue metrics by customer segment
- Churn rate analysis and trends
- Customer demographic breakdown

#### Layer 2: Predictive Insights
- **Uplift Model:** T-Learner (Random Forest)
  - CATE (Conditional Average Treatment Effect) scores
  - Customer segmentation (Sure Things → Persuadables → Lost Causes → Sleeping Dogs)
  - Uplift distribution visualization

- **Churn Model:** XGBoost Classifier
  - ROC-AUC: **0.89**
  - Feature importance via SHAP values
  - Top churn driver: `promo_dependency`

#### Layer 3: Prescriptive Decision Engine
- **Interactive Budget Simulator:**
  - Adjust promotional budget allocation %
  - Real-time P&L calculations
  - Annual impact projections
  - Strategy effectiveness rating (1-10)

---

## 🛠️ Technology Stack

### Machine Learning (Google Colab)
| Component | Tool | Performance |
|-----------|------|-------------|
| **Uplift Model** | scikit-learn (Random Forest) | Estimates treatment effects |
| **Churn Model** | XGBoost | ROC-AUC = 0.89 |
| **Explainability** | SHAP | Feature importance ranking |
| **Data Processing** | Pandas, NumPy | Feature engineering |

### Dashboard Development
| Technology | Version | Purpose |
|-----------|---------|----------|
| **HTML5** | - | Semantic page structure |
| **CSS3** | - | Responsive styling & animations |
| **JavaScript** | ES6+ | Interactive simulator logic |
| **Chart.js** | 4.4.1 | Real-time data visualizations |

### AI Assistants Used
- 🧠 **Claude (Anthropic)** - Statistical methodology & architecture
- 💬 **ChatGPT (OpenAI)** - Rapid code generation & debugging
- 🔍 **Perplexity AI** - Academic research & literature review
- 🤖 **Comet (Perplexity)** - Web research & documentation
- 🎨 **Gemini (Google)** - Visual design optimization
- ⚡ **Cursor IDE** - AI-accelerated dashboard development

---

## 📈 Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Churn Model ROC-AUC** | 0.89 | Strong predictive power |
| **CATE Range** | -0.3 to +0.8 | Individual treatment effects |
| **Persuadables Segment** | ~15-20% | Highest uplift targets |
| **Dashboard Load Time** | <1 sec | No server required |
| **Browser Support** | All modern | HTML5/ES6+ compatible |

---

## 🚀 Quick Start

### For Judges / Evaluators:
1. **Open Dashboard:** [Entropy-Final.html](./Entropy-Final.html)
2. **Understand Architecture:** Read [HOW_WE_BUILT_IT.md](./HOW_WE_BUILT_IT.md)
3. **Technical Details:** Check [AI_TOOLS_AND_ENVIRONMENT.md](./AI_TOOLS_AND_ENVIRONMENT.md)
4. **Development Process:** See [DEVELOPMENT_TOOLS_WORKFLOW.md](./DEVELOPMENT_TOOLS_WORKFLOW.md)

### For Replication:
```bash
# 1. Google Colab Notebooks (links in docs)
# 2. Data: ~100k transactions, 20k customers
# 3. ML Pipeline: 80-20 train-test split
# 4. Dashboard: No dependencies (pure static HTML)

# Simply open Entropy-Final.html in any modern browser
```

---

## 📁 Repository Structure

```
DataVerse/
├── Entropy-Final.html                    ← OPEN THIS (Interactive Dashboard)
├── AI_TOOLS_AND_ENVIRONMENT.md          ← Tech stack & library versions
├── DEVELOPMENT_TOOLS_WORKFLOW.md        ← Tools used (Colab, Claude, ChatGPT, etc.)
├── HOW_WE_BUILT_IT.md                  ← Complete build guide for judges
├── README.md                            ← This file
├── LICENSE                              ← GPL-3.0
└── index.html                           ← Alternative entry point
```

---

## 🧠 The T-Learner Architecture (Why It Matters)

### Problem with Standard Models
```python
# ❌ Standard approach (learns correlation)
model = RandomForest(X_all, y_all)
predicted = model.predict(customer_features)
# Result: Predicts "who will buy" - doesn't isolate treatment effect
```

### Our Causal Approach (T-Learner)
```python
# ✅ T-Learner (isolates causation)
model_treatment = RandomForest(X[promoted==1], y[promoted==1])
model_control = RandomForest(X[promoted==0], y[promoted==0])

# Calculate CATE for each customer
CATE = model_treatment.predict_proba(X)[:, 1] - model_control.predict_proba(X)[:, 1]
# Result: "Who will buy BECAUSE of promotion" - treatment effect
```

**Key Insight:** CATE identifies customers where `promotion makes a difference`

---

## 📊 Customer Segmentation Logic

| Segment | μ₀ (No Promo) | μ₁ (With Promo) | Business Action | Budget % |
|---------|---------------|-----------------|-----------------|----------|
| **Sure Things** | High (>0.7) | High (>0.7) | Don't waste budget | 0-5% |
| **Persuadables** | Low (<0.5) | High (>0.6) | **TARGET HERE** | 60-70% |
| **Lost Causes** | Low (<0.3) | Low (<0.4) | Don't promote | 0% |
| **Sleeping Dogs** | High (>0.6) | Low (<0.5) | Avoid promotion | 0% |

---

## 💡 Judge's Technical Pitch

> "We implemented a **T-Learner causal inference model** using Random Forest to estimate individual treatment effects (CATE). Unlike standard predictive models that identify likely buyers, we identify **Persuadables**—customers whose behavior changes due to promotions. Combined with an **XGBoost churn predictor** (ROC-AUC 0.89) and an **interactive prescriptive dashboard**, we transform ML outputs into actionable financial decisions."

---

## 🎯 Key Findings

✅ **Actionable Segmentation:** Only ~15-20% of customers are Persuadables (high-impact targets)  
✅ **Churn Predictor:** promo_dependency is the strongest churn driver  
✅ **ROI Optimization:** Reallocating 60-70% budget to Persuadables can increase annual revenue by 22-28%  
✅ **Risk Mitigation:** Avoiding promotion to Sleeping Dogs prevents margin erosion  
✅ **Interpretable AI:** SHAP values explain every prediction transparently  

---

## 🔧 Methodology

### Data Preparation (Week 1-2)
1. Load transaction data (~100k records)
2. Feature engineering (promo_dependency, purchase_frequency, membership_duration, support_calls)
3. Train-test split (80-20)
4. Handle class imbalance for churn prediction

### Model Training (Week 2-3)
1. **T-Learner Uplift Model**
   - Train treatment model (μ₁) on promoted customers
   - Train control model (μ₀) on non-promoted customers
   - Calculate CATE = μ₁(x) - μ₀(x) for each customer

2. **Churn Prediction**
   - XGBoost with class weighting
   - Feature importance via SHAP
   - ROC-AUC validation

### Dashboard Development (Week 3-4)
1. Create interactive HTML interface
2. Integrate Chart.js for visualizations
3. Build JavaScript simulator engine
4. Real-time P&L calculations
5. Testing & optimization

---

## 📞 How to Use the Dashboard

1. **Open** `Entropy-Final.html` in your browser
2. **View Metrics:** Current revenue, churn rates, segment sizes
3. **Explore Visualizations:** CATE distribution, ROC curve, SHAP importance
4. **Adjust Sliders:** Budget allocation to different segments
5. **See Results:** Real-time revenue & P&L projections
6. **Rate Strategy:** System gives effectiveness score (1-10)

---

## 👥 Team Members

**Team Entropy**
- Lead Data Scientist: Harnitya Narola
- ML Engineer: [Team Member]
- Dashboard Developer: [Team Member]
- Research Lead: [Team Member]

---

## 📚 Documentation

- **[AI_TOOLS_AND_ENVIRONMENT.md](./AI_TOOLS_AND_ENVIRONMENT.md)** - Complete tech stack documentation
- **[DEVELOPMENT_TOOLS_WORKFLOW.md](./DEVELOPMENT_TOOLS_WORKFLOW.md)** - Development process using Colab, Claude, ChatGPT, Cursor, etc.
- **[HOW_WE_BUILT_IT.md](./HOW_WE_BUILT_IT.md)** - Step-by-step build guide for judges

---

## 🔐 License

GPL-3.0 License - See [LICENSE](./LICENSE) file

---

## 🌐 Project Links

- 📊 **Dashboard:** [Entropy-Final.html](./Entropy-Final.html)
- 💻 **GitHub:** https://github.com/Harnitya29/DataVerse
- 🎯 **Competition:** DataVerse Datathon 2026
- 📅 **Submission Date:** February 27, 2026

---

## 🚀 Deployment

Dashboard is deployed via GitHub Pages:  
https://harnitya29.github.io/DataVerse/

---

**Last Updated:** February 27, 2026 at 7 PM IST  
**Status:** ✅ Complete and Ready for Submission

---

## ❓ FAQ

**Q: Why T-Learner and not standard models?**  
A: T-Learner isolates causal effects (treatment effect = outcome difference between promoted and non-promoted). Standard models only learn correlation.

**Q: How accurate is the churn model?**  
A: ROC-AUC = 0.89 on test set. Excellent discriminative power.

**Q: Can this scale to production?**  
A: Dashboard: Yes (static HTML). ML pipeline: Would need Docker + FastAPI for real-time predictions.

**Q: How were the ML models trained?**  
A: Google Colab notebooks with scikit-learn, XGBoost, SHAP. Reproducible pipeline.

**Q: What data is needed?**  
A: Transaction records + customer demographics + promotion flags. Works with any e-commerce dataset.

---

*Built with ❤️ by Team Entropy using Google Colab, Claude, ChatGPT, Cursor IDE, and modern web technologies*
