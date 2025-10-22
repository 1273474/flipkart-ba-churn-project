# Proactive Customer Churn Intervention Strategy

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/pandas-2.0-blue.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)

This end-to-end data science project moves from raw, multi-table e-commerce data to a complete, ROI-positive business strategy. The goal is to not only *predict* customer churn but also to identify the most *profitable* customers to target with an intervention campaign.

This project was built to demonstrate the end-to-end lifecycle of a Business Analyst, from data extraction and feature engineering to predictive modeling, causal inference, and final strategic recommendation.

## 🚀 The Core Business Problem

The Olist e-commerce platform, like many marketplaces, has low customer repeat rates. The business wants to run a marketing campaign (e.g., a 20% off coupon) to prevent churn and increase customer lifetime value.

This project answers two critical questions:
1.  **Prediction:** Which of our valuable, active customers are at the highest risk of churning?
2.  **Prescription:** Which of these "at-risk" customers should we *actually* send a coupon to in order to maximize our profit and avoid wasting money on customers who would churn anyway ("Lost Causes") or who would return anyway ("Sure Things")?

## 📊 Methodology: The 8-Step Analytics Pipeline

This project is broken down into a single notebook (`Customer_Churn_Analysis.ipynb`) that follows a complete BA workflow.

### Step 1: Data Ingestion & Cleaning
* Loaded 8 raw `.csv` files (e.g., `orders`, `customers`, `payments`, `reviews`, etc.) into Pandas DataFrames.
* Merged all tables into a single master DataFrame, handling duplicates and cleaning data types.

### Step 2: Problem Framing & EDA
* **Crucial Insight:** Initial analysis showed that any "churn" definition applied to the *entire* customer base resulted in a 95%+ class imbalance. This makes modeling useless (as proven by initial failed models).
* **The Pivot:** Re-framed the business problem to focus on a high-value, solvable cohort: **"Of all customers who made 2 or more purchases in 2017, which ones will return for another purchase in the next 90 days?"**
* This created a smaller, more valuable cohort with a balanced and realistic target variable.

### Step 3: Feature Engineering
* Built a "customer features" table based on the 2017 cohort's historical data.
* Engineered key features like `frequency`, `monetary`, `avg_review_score`, `avg_installments`, and `favorite_category`.
* Created the final target label: `returned_in_90d` (1 = Yes, 0 = No).

### Step 4: Customer Segmentation
* Used **K-Means Clustering** on the RFM features to identify 4 distinct customer personas:
    * **Champions:** High-value, loyal customers.
    * **At-Risk VIPs:** High-value customers who haven't shopped in a while.
    * **New Customers:** Low-value, new repeat buyers.
    * **Lost Customers:** Churned repeat buyers.

### Step 5: Churn Prediction Model
* Trained a **Logistic Regression** model (with `class_weight='balanced'`) to predict `returned_in_90d`.
* **Result:** The model achieved an **Average Precision (PR-AUC) of 0.1027**, which was **3x better** than the "No Skill" baseline (0.0342).
* **Key Insight:** Model interpretability showed that **`favorite_category` was the #1 predictor of loyalty.**
    * **Loyal Categories:** `auto`, `electronics`, `watches_gifts`
    * **Churn-Prone Categories:** `health_beauty`, `fashion_bags_accessories`

### Step 6: Uplift Modeling (Causal Inference)
* **Challenge:** The dataset had no A/B test data (no "treatment" vs. "control" groups).
* **Solution:** **Simulated a historical campaign** based on the insight from Step 5. We assumed a coupon would have a 30% chance of "flipping" a churn-prone customer, but no effect on a loyal customer.
* Built a **T-Learner (Two-Model) uplift model** to isolate the causal *effect* of the coupon.
* This model generated an `uplift_score` for every customer, answering the question: "How much more likely is this customer to return *if* we send them a coupon?"

### Step 7: ROI Simulation
* This is the "money" step. I built a profit simulator based on the model's outputs.
* **Assumptions:**
    * `Avg. Customer Value (CLV proxy)`: $469.05 (mean 2017 spend)
    * `Cost Per Intervention (20% Coupon)`: $44.58 (20% of AOV)
* The model calculated the `expected_profit` for targeting every single customer.
* **The Strategy:** By targeting only the **333 customers** with a positive expected profit, the campaign would generate **$49,020 in net profit** from a cost of $14,845.
* **Final Result:** A data-driven, actionable strategy with a **330.21% ROI**.

### Step 8: A/B Test Design
* Wrote a formal experiment design document to validate this model in production.
* **Hypothesis:** Targeting the 333 "persuadable" customers will result in a statistically significant lift in 90-day return rate and a net positive profit.
* **Groups:** Group A (Control, no coupon) vs. Group B (Treatment, 20% coupon).
* **Primary Metric:** `90-day Return Rate`.
* **Secondary Metrics:** `Average Order Value` (AOV) and `Net Profit Per Customer`.

## 🔑 Key Insights
1.  **Product Category is the #1 Predictor:** A customer's loyalty is best predicted by *what* they buy. "Hobby" categories (e.g., `auto`) create loyal customers, while "commodity" categories (e.g., `health_beauty`) create one-time shoppers.
2.  **Churn Prediction is Not Enough:** A simple churn model wastes money. The **Uplift Model** was essential to identify the "Persuadables" and avoid targeting "Sure Things" (who would return anyway) and "Lost Causes" (who would churn anyway).
3.  **Data-Driven Profit:** The final model isn't just an accuracy score; it's a profit-generating engine. We identified a 333-customer segment that would deliver a **330.21% ROI** on a targeted marketing campaign.

## 🛠️ How to Run This Project
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/flipkart-ba-churn-project.git
    cd flipkart-ba-churn-project
    ```
2.  **Set up the environment:**
    * Upload the `Customer_Churn_Analysis.ipynb` notebook to Google Colab.
    * The notebook's first cell will install all required libraries (`kaggle`, `causalml`, etc.).
3.  **Get Kaggle API Key:**
    * Follow the instructions in the notebook (Step 1.B.1) to download your `kaggle.json` API key.
4.  **Run the Notebook:**
    * Run the cells in order. The notebook will automatically download the data, perform all analysis, and build all models.

## 📦 Libraries Used
* `pandas`
* `numpy`
* `matplotlib` / `seaborn`
* `scikit-learn` (for data processing, K-Means, Logistic Regression, and the T-Learner)
* `lightgbm` & `shap` (used for initial modeling before pivoting to the final, simpler model)
* `kaggle` (for data ingestion)
