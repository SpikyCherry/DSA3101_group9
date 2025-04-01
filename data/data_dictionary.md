# 📘 Data Dictionary: Credit Card Customers Dataset

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
