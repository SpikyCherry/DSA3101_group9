# 📘 Data Dictionary 1: Credit Card Customers Dataset

| Column Name              | Data Type   | Description                                    | Possible Values / Range              |
|:-------------------------|:------------|:-----------------------------------------------|:-------------------------------------|
| CLIENTNUM                | integer     | Unique identifier for each customer            | Numeric ID                           |
| Attrition_Flag           | categorical | Customer attrition status                      | Existing Customer, Attrited Customer |
| Customer_Age             | integer     | Age of the customer                            | 26–73                                |
| Gender                   | categorical | Gender of the customer                         | M, F                                 |
| Dependent_count          | integer     | Number of dependents                           | 0–5                                  |
| Education_Level          | categorical | Highest education level achieved               | High School, Graduate, etc.          |
| Marital_Status           | categorical | Marital status of the customer                 | Married, Single, Divorced            |
| Income_Category          | categorical | Estimated annual income                        | Less than $40K, $40K–$60K, etc.      |
| Card_Category            | categorical | Type of credit card                            | Blue, Silver, Gold, Platinum         |
| Months_on_book           | integer     | Duration of relationship with the bank         | 13–56                                |
| Total_Relationship_Count | integer     | Total number of products held by the customer  | 1–6                                  |
| Months_Inactive_12_mon   | integer     | Inactive months in the last 12 months          | 0–6                                  |
| Contacts_Count_12_mon    | integer     | Number of contacts in the last 12 months       | 0–6                                  |
| Credit_Limit             | float       | Credit limit assigned to the customer          | $1438–$34516                         |
| Total_Revolving_Bal      | integer     | Total revolving balance on the card            | 0–2517                               |
| Avg_Open_To_Buy          | float       | Average available credit to spend              | 0–34516                              |
| Total_Trans_Amt          | integer     | Total transaction amount in the last 12 months | 510–18484                            |
| Total_Trans_Ct           | integer     | Total transaction count in the last 12 months  | 10–139                               |
| Avg_Utilization_Ratio    | float       | Average utilization ratio of the credit limit  | 0.0–0.999                            |

---

## 📝 Notes

- **Attrition_Flag** is the target variable for this dataset, indicating whether the customer has churned (account closed). The goal is to predict this variable based on the other features.
- The dataset contains a mixture of **demographic** and **product-related** features:
  - **Demographic Variables**: `Customer_Age`, `Gender`, `Dependent_Count`, `Education_Level`, `Marital_Status`, `Income_Category` are all categorical or numerical variables describing the customer's personal information.
  - **Product-Related Variables**: `Card_Category`, `Months_on_book`, `Total_Relationship_Count`, `Credit_Limit`, `Total_Revolving_Bal`, `Avg_Open_To_Buy`, `Total_Trans_Amt`, `Total_Trans_Ct`, `Avg_Utilization_Ratio` are related to the customer's interactions with the bank and credit card usage.
- **Numerical Variables**: `Customer_Age`, `Dependent_Count`, `Months_on_book`, `Total_Relationship_Count`, `Credit_Limit`, `Total_Revolving_Bal`, `Avg_Open_To_Buy`, `Total_Amt_Chng_Q4_Q1`, `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`, and `Avg_Utilization_Ratio` are continuous variables that might need normalization or standardization.
- **Categorical Variables**: `Gender`, `Education_Level`, `Marital_Status`, `Income_Category`, `Card_Category` are categorical variables that may require encoding (e.g., one-hot encoding) before use in machine learning models.
- The **Income_Category** and **Marital_Status** variables have a category for "Unknown", which may represent missing or unspecified data and could require special handling.
- There might be imbalances in the target variable `Attrition_Flag`, with fewer customers who churn, so techniques like oversampling or class weighting may be necessary for balanced predictions.


# 📘 Data Dictionary 2: Banking Dataset - Marketing Targets
| **Field Name**         | **Data Type**          | **Description**                                                                                              | **Example**                                                                                                     |
|------------------------|------------------------|--------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| age                    | Integer                | Age of the customer                                                                                           | 30                                                                                                              |
| job                    | String (Categorical)   | Type of job                                                                                                   | "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services" |
| marital                | String (Categorical)   | Marital status                                                                                                 | "married", "divorced", "single"; note: "divorced" means divorced or widowed                                    |
| education              | String (Categorical)   | Education level                                                                                                | "primary", "secondary", "tertiary"                                                                              |
| default                | Binary                 | Has credit in default?                                                                                         | "yes", "no"                                                                                                     |
| balance                | Integer                | Average yearly balance, in euros                                                                               | 1787                                                                                                            |
| housing                | Binary                 | Has housing loan?                                                                                              | "yes", "no"                                                                                                     |
| loan                   | Binary                 | Has personal loan?                                                                                             | "yes", "no"                                                                                                     |
| contact                | String (Categorical)   | Contact communication type                                                                                     | "unknown", "telephone", "cellular"                                                                              |
| day                    | Integer                | Last contact day of the month                                                                                   | 27                                                                                                              |
| month                  | Integer                | Last contact month of the year                                                                                  | 4                                                                                                               |
| duration               | Integer                | Last contact duration, in seconds                                                                               | 20                                                                                                              |
| campaign               | Integer                | Number of contacts performed during this campaign and for this client, includes last contact                   | 1                                                                                                               |
| pdays                  | Integer                | Number of days that passed by after the client was last contacted from a previous campaign (-1 means client was not previously contacted) | 2                                                                                                               |
| previous               | Integer                | Number of contacts performed before this campaign and for this client                                          | 1                                                                                                               |
| poutcome               | String (Categorical)   | Outcome of the previous marketing campaign                                                                     | "unknown", "other", "failure", "success"                                                                        |
| y                      | Binary                 | Has the client subscribed to a term deposit?                                                                    | "yes", "no"                                                                                                     |
| conversion_rate        | Float                  | Success rate per contact attempt                                                                                | 0.50                                                                                                            |
| best_contact_time      | String                 | Optimal contact time based on job type                                                                          | "6-8pm", "12-2pm", "2-4pm", "4-5pm"                                                                             |
| fatigue_score          | Float                  | Score measuring campaign fatigue                                                                                | 0.10                                                                                                            |
| deposit_amount         | Float                  | Actual deposit amount (y=1) or campaign recommended amount (y=0)                                                | 39000, 1257000                                                                                                  |
| term                   | String (Categorical)   | Actual deposit term (y=1) or campaign recommended term (y=0)                                                   | "current", "three_months", "six_months", "one_year", "two_year"                                                |
| interest_rate          | Float                  | Actual deposit interest rate (y=1) or campaign recommended interest rate (y=0), based on real historical background and ECB data | 1.50, 2.10                                                                                                      |
| cost                   | Float                  | Estimated marketing expense per customer, calculated based on call duration, number of contact attempts, and an assumed telemarketing rate of USD 38 per hour | 6.016667, 0.886667                                                                                              |
| purchase_frequency     | Integer                | How often a customer responded positively across two campaigns, calculated by summing successful outcomes from the previous campaign (poutcome) and the current campaign (y) | 0, 1, 2                                                                                                         |
| revenue                | Float                  | The income from a single successful purchase in the current campaign, calculated as deposit_amount × interest_rate when the customer subscribes (y = 1) | 1963845.0, 0.0                                                                                                  |
| customer_lifespan      | Integer                | Estimates how many more days a customer is expected to stay engaged, calculated based on average retirement and life expectancy | 16060, 15330                                                                                                    |
| CLV                    | Float                  | Customer Lifetime Value, estimates the total revenue a customer is expected to generate over their entire relationship with the business | 5.868357e+10, 3.177508e+09, 0.0                                                                                |
| new_customer           | Binary                 | Someone whose previous campaign outcome was unsuccessful but who subscribed in the current campaign             | True, False                                                                                                     |
| CAC                    | Float                  | Customer Acquisition Cost, the average cost to acquire a new customer, typically calculated at the group level (e.g., by contact type), and assigned to each row | 168.21091, 56.819262                                                                                           |
| ROI                    | Float                  | Return on Investment, calculated as CLV/CAC, indicating the value generated for each dollar spent on acquiring a customer | 1.059702e+09, 0.0                                                                                              |


## 📝 Notes

**y(subscription)** is the target variable for this dataset, indicating whether the client subscribed to a term deposit. The goal is to predict this variable based on the other features.
The dataset contains a mixture of demographic and product-related features:
**Demographic Variables**: age, education, job, marital are categorical variables that describe the clients.
Financial Variables: 'default', 'balance', 'housing', 'loan' are continuous variables that might need normalization or standardization.
**Behaviour Variables**: campaign, contact, poutcome, pdays, previous are categorical variables that may require encoding (e.g., one-hot encoding) before use in machine learning models.
Numerical variables: age, day, duration, campaign, pdays, previous are discrete and balance is continuous.
Generated Variables: conversion_rate, fatigue_score, deposit_amount, interest_rate are numerical variables, best_contact_time, term are categorical variables.
Synthetic variables: 
deposit_amount: The amount of the term deposit if the customer purchased (y=1). Or the amount of term deposit recommended by the campaign initially, but was rejected(y=0) 
term: The term of the purchased deposit(y=1) or the initial recommended term(y=0). It has categories "current", "three_months", "six_months", "one_year" and "two_year". They were determined by the real business background when the dataset was collected.
interest_rate: The interest rate of the purchased deposit(y=1) or the initial interest rate recommended by the campaign, but was rejected(y=0). The values correspond to the term and were determined based on the real business background when the dataset was collected, and the ECB data and reports.
		
The job and Marital_Status variables have a category for "Unknown", which may represent missing or unspecified data and could require special handling.
There might be imbalances in the target variable y(subscription , with fewer customers who churn, so techniques like oversampling or class weighting may be necessary for balanced predictions.
