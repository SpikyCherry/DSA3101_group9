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