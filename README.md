# 🧠 Machine Learning for Personalized Marketing Campaigns in Banking

This repository contains a collaborative data science project aimed at helping a bank optimize its marketing strategy using machine learning. The goal is to develop personalized, data-driven marketing campaigns that increase customer engagement, conversion rates, and marketing ROI.

A team of 10 data scientists collaborated to explore various supervised and unsupervised learning tasks using real-world and synthetic customer data.

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/                      # Place manually downloaded datasets here
│   ├── synthesized/              # Synthetic data generated for specific tasks
│   ├── processed/                # Cleaned data saved by each question
│   ├── generate_synthetic_data.py  # Script to generate synthetic data
│   ├── data_dictionary.md        # Descriptions of dataset features
│   └── data_instructions.md      # Instructions to download/generate data
│
├── utils/                        # Shared functions (e.g., encoding)
│
├── models/
│   ├── question_1/               # Trained models saved per question
│   ├── question_2/
│   └── ...
│
├── EDA/                          # Exploratory notebooks for global understanding
│
├── question_x/
│   ├── preprocess.py             # Load and clean data for this question
│   ├── train_model.py            # Final model implementation
│   └── analysis.ipynb            # Full workflow: EDA, training, visualization
```

---

## 🔄 Project Workflow

1. **Data Collection**
   - Three real-world datasets from Kaggle are used.
   - Synthetic data is generated to augment the dataset where necessary.

2. **Preprocessing**
   - Shared and question-specific transformations are applied.
   - Final datasets are saved in `data/processed/`.

3. **Modeling**
   - Each question runs a separate training pipeline using their preprocessed dataset.
   - Trained models are saved under `models/question_x/`.

4. **Notebook Reporting**
   - Each `analysis.ipynb` provides an end-to-end record: data exploration, model comparisons, visualization, and insights.

---

## 📚 Data Access and Documentation

- 📄 [Data Download & Generation Instructions](data/data_instructions.md)
- 📘 [Data Dictionary](data/data_dictionary.md)

Please follow these documents **before running any scripts**.

---

## 📊 Dataset Variable Description

The Bank Churn dataset consists of customer information, which is used to predict whether a customer will leave the bank (i.e., churn). Below is a description of each variable in the dataset.

| **Variable**                | **Description**                                                                                           |
|-----------------------------|-----------------------------------------------------------------------------------------------------------|
| `CLIENTINUM`                 | Client number. Unique identifier for the customer holding the account.                                   |
| `Attrition_Flag`             | Internal event (customer activity) variable - if the account is closed then 1 else 0                     |
| `Customer_Age`               | Demographic variable - Customer's Age in Years                                                           |
| `Gender`                     | Demographic variable - M=Male, F=Female                                                                  |
| `Dependent_Count`            | Demographic variable - Number of dependents                                                              |
| `Education_Level`            | Demographic variable - Educational Qualification of the account holder (example: high school, college graduate, etc.)   |
| `Marital_Status`             | Demographic variable - Married, Single, Divorced, Unknown                                                       |
| `Income_Category`            | Demographic variable - Annual Income Category of the account holder (< $40K, $40K - 60K, $60K - $80K, $80K-$120K, > $120K, Unknown)    |
| `Card_Category`              | Product Variable - Type of Card (Blue, Silver, Gold, Platinum)                                              |
| `Months_on_book`             | Period of relationship with bank                                                                            |
| `Total_Relationship_Count`   | Total no. of products held by the customer                                                                |
| `Months_Inactive_12_mon`     | No. of months inactive in the last 12 months                                                              |
| `Contracts_Count_12_mon`     | No. of Contacts in the last 12 months                                                                     |
| `Credit_Limit`               | Credit Limit on the Credit Card                                                                            |
| `Total_Revolving_Bal`        | Total Revolving Balance on the Credit Card                                                                 |
| `Avg_Open_To_Buy`            | Open to Buy Credit Line (Average of last 12 months)                                                        |
| `Total_Amt_Chng_Q4_Q1`       | Change in Transaction Amount (Q4 over Q1)                                                                  |
| `Total_Trans_Amt`            | Total Transaction Amount (Last 12 months)                                                                  |
| `Total_Trans_Ct`             | Total Transaction Count (Last 12 months)                                                                   |
| `Total_Ct_Chng_Q4_Q1`        | Change in Transaction Count (Q4 over Q1)                                                                   |
| `Avg_Utilization_Ratio`      | Average Card Utilization Ratio                                                                             |

---

## 📝 Reproducibility

To run a specific question:

```bash
python question_x/preprocess.py
python question_x/train_model.py
# or open question_x/analysis.ipynb
```

