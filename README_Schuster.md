# 💳 E-Commerce & Retail Case Study: Schuster Vendor Payment Intelligence
**K-Means Clustering · Random Forest · GridSearchCV · Feature Engineering · Accounts Receivable Analytics**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

---

## 📌 Project Overview

Schuster, a dominant global retailer in sports goods and accessories, faced a critical operational challenge: **widespread vendor payment delays** that were disrupting cash flow, straining supplier relationships, and limiting the collection team's ability to prioritize work effectively.

This project delivers a two-stage ML solution — first **segmenting customers by payment behavior** using K-Means clustering, then **predicting late payments on open invoices** using a tuned Random Forest classifier. The output enables the collections team to intervene proactively, focusing energy on the highest-risk accounts before payments become overdue.

**Key outcome:** The Random Forest model achieves **81% precision in identifying late payments**, enabling targeted, data-driven collection strategy across Schuster's global vendor portfolio.

---

## 📂 Repository Structure

```
E-Commerce_Retail_Schuster/
│
├── E-Commerce_and_Retail_-__Case_Study_Final.ipynb   # Full ML pipeline (EDA → Segmentation → Prediction → Deployment)
├── Received_Payments_Data.csv                         # Historical invoice payment data
├── Open_Invoice_data.csv                              # Current open invoices for scoring
├── Data_Dictionary.xlsx                               # Field definitions and schema reference
├── E-commerce_and_Retail_Case_Study.pdf               # Stakeholder presentation deck
└── README.md
```

---

## 🧠 Problem Statement

| Challenge | Detail |
|---|---|
| Client | Schuster — global sports goods retailer |
| Core issue | Vendor payment delays impacting operational efficiency and financial stability |
| Business need | Segment customers by payment risk; predict late payments on open invoices |
| Collection team need | Prioritize follow-ups proactively, not reactively |
| Top 10 customers | Hold **94% of total payment receivables** — concentration risk |

The dataset consisted of two files: **Received Payments Data** (historical closed invoices used for training) and **Open Invoice Data** (live invoices for prediction/deployment).

---

## 🔬 Technical Approach

### 1. Data Cleaning & Feature Engineering

**Received Payments Dataset:**
- Removed invoices with `USD Amount < 0` (invalid entries)
- Converted date columns (`RECEIPT_DATE`, `INVOICE_CREATION_DATE`, `DUE_DATE`) to datetime format
- Engineered the **target variable**: `Payment Status = "Late Payment"` where `RECEIPT_DATE > DUE_DATE`, else `"On-time Payment"`
- Engineered `Payment_Terms`: difference in days between `DUE_DATE` and `INVOICE_CREATION_DATE`
- Engineered `Average_payment_time`: difference in days between `RECEIPT_DATE` and `INVOICE_CREATION_DATE`
- Applied **percentile-based outlier removal**: retained `Average_payment_time` within the 1st–99th percentile range (-5 to 1,050 days) and `USD Amount` up to the 99.5th percentile ($6.3M)
- **Class imbalance noted**: ~66% of invoices in historical data were late payments

**Open Invoice Dataset:**
- Converted date columns and computed `PAYMENT_TERM` (days from invoice creation to due date)
- Computed `Average_payment_time` as days remaining between `AS_OF_DATE` and `Due Date`
- Removed negative AGE records and non-positive USD amounts
- Applied `MinMaxScaler` to numerical features (`USD Amount`, `Payment_Terms`) to match the training pipeline

---

### 2. Exploratory Data Analysis (EDA)

**Currency & Invoice Type Analysis:**
- **SAR (Saudi Riyal)** was the largest contributor to late payments, followed by USD and AED — actionable signal for prioritizing collection strategies by region/currency.
- **Goods invoices** were the dominant source of late payments compared to Non-Goods, supporting a recommendation to concentrate collection efforts on goods-type transactions.

**Payment Terms Analysis:**
- Payment terms of **60 days and 30 days from invoice creation date** were associated with the highest late payment volumes.
- Terms structured from the **End of Month (EOM)** — specifically 15 and 60 days EOM — produced more reliable on-time payments.
- Recommendation: introducing a **grace period buffer** reduces last-minute pressure on both customers and invoicing teams.

**Weekly Seasonality:**
- Late payments exhibit clear **weekly seasonality** — specific weeks within months show pronounced spikes (e.g., week of Jan 24, 2021 reached ~7,000 late payments).
- Gives collections teams a predictive calendar for when to pre-emptively escalate outreach.

**Customer Concentration (Pareto Effect):**
- Top 10 customers — SEPH Corp, FARO Corp, PARF Corp, ALLI Corp, AREE Corp, HABC Corp, RADW Corp, L OR Corp, CGR Corp, PCD Corp — account for **94% of total receivables**.
- Average late payment time varied significantly: FARO Corp averaged ~175 days late; CGR Corp averaged ~35 days late — requiring differentiated treatment strategies per customer.

---

### 3. Customer Segmentation — K-Means Clustering

**Features used:**
- `Mean_Payment_time` — average days a customer takes to pay across all invoices
- `Std_Dev_Payment_time` — variability (consistency) of their payment behavior

**Methodology:**
- Aggregated individual invoice-level data to customer-level summary statistics using `groupby`
- Null `Std_Dev` values (single-invoice customers) imputed with 0
- Applied `StandardScaler` before clustering to normalize scale
- Used **Silhouette Score analysis** across k = 2 to 8 to select optimal cluster count

**Optimal k = 3** selected based on silhouette score plateau and business interpretability.

**Payer Segments Identified:**

| Cluster | Segment Name | Mean Payment Time | Std Dev | Behavior |
|---|---|---|---|---|
| Cluster 0 | **Punctual Payers** | Low | Low | Reliable, consistent on-time payment |
| Cluster 1 | **Erratic Payers** | High | Low | Consistently late but predictably so |
| Cluster 2 | **Variable Payers** | High | High | Unpredictable; both late and high variance |

**Cluster assignment was then mapped back** to each invoice in the received payments dataset as a new feature (`cluster_id`) for use in predictive modeling.

**Segmentation was also applied independently to the Open Invoice dataset** using the same K-Means pipeline (k=3, StandardScaler) — ensuring open invoice customers received a cluster label for prediction.

---

### 4. Predictive Modeling — Random Forest Classifier

**Features used in the final model:**
- `USD Amount` (MinMax scaled)
- `Payment_Terms` (MinMax scaled)
- `cluster_id_1` (dummy: Cluster 1 membership)
- `cluster_id_2` (dummy: Cluster 2 membership)

**Target variable:** `Late payment` (binary: 1 = Late, 0 = On-time)

**Train/Test Split:** 70% train / 30% test (`random_state=100`)

**Model:** `RandomForestClassifier` with hyperparameter tuning via `GridSearchCV` (5-fold CV)

**GridSearchCV parameter grid:**

```python
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
```

**Model Performance:**

| Metric | Late Payment (Class 1) | On-time Payment (Class 0) |
|---|---|---|
| Precision | **81%** | 48% |
| Recall | **60%** | 73% |
| F1-Score | **69%** | 58% |
| **Overall Accuracy** | **64%** | — |

The model is **optimized for Late Payment detection** — the operationally critical class. 81% precision on late payments means that when the model flags an invoice as high-risk, it is correct the large majority of the time, enabling confident prioritization by the collections team.

---

### 5. Model Deployment — Open Invoice Scoring

The tuned Random Forest model was applied directly to the **Open Invoice dataset** to generate late payment predictions on all currently outstanding invoices:

- Open invoice customers were re-segmented using the K-Means pipeline
- Cluster labels were dummy-encoded and mapped to match training feature names
- `USD Amount` and `Payment_Terms` were scaled using `MinMaxScaler`
- `tuned_clf.predict(open_invoices_test)` generated binary predicted labels for each open invoice
- Output: a prioritized list of invoices predicted as high-risk for late payment

---

## 📊 Key Insights & Business Recommendations

### Collection Strategy by Segment

**Punctual Payers — "Reliable Champions"**
Maintain positive relationships. Offer early payment discounts, loyalty rewards, referral incentives, and exclusive access for consistent on-time behavior. Recognize and celebrate good payers publicly.

**Erratic Payers — "Steady Improvement Focus"**
Establish explicit payment term agreements. Offer incremental incentives tied to improvement milestones. Provide AR support to help customers adhere to timelines.

**Variable Payers — "Adaptive Engagement"**
Implement flexible payment scheduling. Deploy personalized follow-ups to surface cash flow constraints. Tailor solutions per customer — these accounts require the most active collector attention.

### Invoice & Currency Prioritization

- Prioritize **Goods invoices** first — they represent the majority of late payment volume.
- Flag invoices denominated in **SAR, USD, and AED** for earlier follow-up.
- Consider restructuring payment terms toward **EOM-anchored schedules** (15 or 60 days EOM) to improve on-time compliance.

### Seasonal Calendar Planning

Collections teams should pre-load capacity in historically high-risk weeks (late January, mid-April, late May) to handle elevated late payment volumes proactively rather than reactively.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Manipulation | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly Express |
| Machine Learning | scikit-learn (RandomForestClassifier, KMeans, GridSearchCV, MinMaxScaler, StandardScaler, silhouette_score) |
| Model Evaluation | Classification Report, Accuracy Score, Precision, Recall, F1-Score |
| Notebook | Jupyter Notebook |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly jupyter
```

### Run the Notebook

```bash
git clone https://github.com/smtypradhan/<repo-name>.git
cd <repo-name>
jupyter notebook "E-Commerce_and_Retail_-__Case_Study_Final.ipynb"
```

Ensure `Received_Payments_Data.csv` and `Open_Invoice_data.csv` are in the same directory as the notebook before running.

---

## 📈 Results Summary

- ✅ Segmented Schuster's full customer base into **3 actionable payment behavior clusters** (Punctual, Erratic, Variable) using K-Means with silhouette-score-guided k selection
- ✅ Built a **Random Forest classifier** with GridSearchCV tuning achieving **81% precision on late payment detection**
- ✅ Identified that the **top 10 customers hold 94% of receivables** — enabling a focused, high-impact collection strategy
- ✅ Discovered **weekly seasonality** in late payments — enabling predictive capacity planning for collection teams
- ✅ Deployed the model against **live open invoice data** to generate actionable risk scores for current outstanding balances
- ✅ Delivered segment-specific playbooks linking ML output directly to collection tactics

---

## 👤 Author

**Smriti Pradhan**
MS Data Analytics | Touro Graduate School of Technology
[LinkedIn](https://linkedin.com) · [GitHub](https://github.com/smtypradhan)

---

## 📄 License

This project is for educational and portfolio purposes.
