import pandas as pd
import numpy as np
import os

# Interest rates from 2008/7 to 2010/12, estimated from ECB
rate_table=[
    {"time":"2008/7","MRO": 4.25,"current": 0.75,"three_months": 3.20,"six_months": 3.60,"one_year": 3.85,"two_year": 4.05},
    {"time":"2008/12","MRO": 2.50,"current": 0.40,"three_months": 2.10,"six_months": 2.50,"one_year": 2.75,"two_year": 3.00},
    {"time":"2009/5","MRO": 1.00,"current": 0.30,"three_months": 1.20,"six_months": 1.50,"one_year": 1.70,"two_year": 2.00},
    {"time":"2009/12","MRO": 1.00,"current": 0.25,"three_months": 1.00,"six_months": 1.25,"one_year": 1.40,"two_year": 1.70},
    {"time":"2010/6","MRO": 1.00,"current": 0.20,"three_months": 0.90,"six_months": 1.10,"one_year": 1.25,"two_year": 1.55},
    {"time":"2010/12","MRO": 1.00,"current": 0.20,"three_months": 0.85,"six_months": 1.00,"one_year": 1.20,"two_year": 1.50}
]

# Deposit terms
deposit_products=["current","three_months","six_months","one_year","two_year"]

np.random.seed(369)

def generate_deposit_features(y_value):
    if str(y_value).strip().lower()=="yes":
        deposit_amount=np.random.choice(np.arange(1000,500000+1,100)) # Deposit range
        rate_row=np.random.choice(rate_table)
        deposit_product=np.random.choice(deposit_products)
        interest_rate=rate_row[deposit_product]
        deposit_term=deposit_product
        return pd.Series([deposit_amount,deposit_term,interest_rate])
    else:
        return pd.Series([0,0,0])
    
# Set your own file directory
file_wd=r"C:\Your_directory"

# Put the train.csv and test.csv in the same directory also    
train_path=os.path.join(file_wd,"train.csv")
test_path=os.path.join(file_wd,"test.csv")
train_new_path=os.path.join(file_wd,"train_new.csv")
test_new_path=os.path.join(file_wd,"test_new.csv")

train=pd.read_csv(train_path,sep=";")
test=pd.read_csv(test_path,sep=";")

train[["deposit_amount","term","interest_rate"]]=train["y"].apply(generate_deposit_features)
test[["deposit_amount","term","interest_rate"]]=test["y"].apply(generate_deposit_features)

train.to_csv(train_new_path,index=False)
test.to_csv(test_new_path,index=False)
