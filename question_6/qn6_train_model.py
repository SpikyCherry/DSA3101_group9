"""

Subgroup B Qn 1 Predicting Customer Preferences/ Qn6

"""
#####################################################
# 1. Pacakages
#####################################################
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
import os
import sys
import joblib

#####################################################
# 2. Data Preprocessing and Feature Engineering
#####################################################

def load_and_preprocess_data(file_path,prediction=False):
    """
    Load the data for preprocessinng and featureing engineering   
    The model may be adapted for other variables or recommendation tasks.
    The changes of variables should be done here
    
    Argument:
        file_path: str, the path of the data file
    return:
        user_features: numpy array, the features extracted from the customer side
        product_features: numpy array, the features extracted from the product/service side
        cross_features: numpy array, the interactions between the customers and the campaign 
        labels: numpy array, label/outcome
    """
    # Read the data in csv
    df=pd.read_csv(file_path)

    #Use oversample to solve the problem of imbalanced data
    train_yes=df[df['y']==1]
    df=pd.concat([df]+[train_yes]*6,ignore_index=True)

    # Labels/Outcomes
    label_y=df['y'].values
    label_term=df['term'].values
    label_ir=df['interest_rate'].values
    
    # Categorize the features (could be adjusted)
    user_feature_cols=['age','job','marital','education','default','balance','housing','loan']
    product_feature_cols=['deposit_amount','interest_rate'] 
    cross_feature_cols=['contact','day','month','duration','campaign','pdays','previous','poutcome']
    
    user_features=df[user_feature_cols].values
    product_features=df[product_feature_cols].values
    cross_features=df[cross_feature_cols].values

    return  user_features,product_features,cross_features,label_y,label_term,label_ir
    
#####################################################
# 3. Data Segmentation
#####################################################

def split_train_val_test(user_feats,product_feats,cross_feats,label_y,label_term,label_ir,train_ratio=0.6,val_ratio=0.2,test_ratio=0.2,random_state=42):
    """
    Split the dataset into training, validation and testing set.
    """
    # need to seperate twice since the default method only support binay split
    X_user_train,X_user_temp,X_product_train, X_product_temp,X_cross_train,X_cross_temp,y_train,y_temp,term_train,term_temp,ir_train,ir_temp=train_test_split(
        user_feats,product_feats,cross_feats,label_y,label_term,label_ir,test_size=(1-train_ratio), random_state=random_state)
    
    val_portion=val_ratio/(val_ratio+test_ratio)

    X_user_val,X_user_test,X_product_val,X_product_test,X_cross_val, X_cross_test,y_val, y_test,term_val, term_test,ir_val, ir_test=train_test_split(
        X_user_temp,X_product_temp,X_cross_temp,y_temp,term_temp,ir_temp,test_size=(1-val_portion),random_state=random_state)
    
    return (X_user_train,X_user_val,X_user_test,X_product_train,X_product_val,X_product_test,X_cross_train,X_cross_val,X_cross_test,y_train,y_val,y_test,
            term_train,term_val,term_test,ir_train,ir_val,ir_test)

#####################################################
# 4. Modified Deep Structured Semantic Models
#####################################################

class ThreeTowerModel(nn.Module):
    def __init__(self,user_feature_dim,product_feature_dim,cross_feature_dim,hidden_dim=10,embedding_dim=4):
        """
        Modified Deep Structured Semantic Models/Three Towers:
          - First tower: User feature tower
          - Second tower: Product/Service feature tower
          - Third tower: Interaction/Campaign feature tower

        Outputs
          - Probability of y=1:Sigmoid, predicting the probability of the custormer buying deposit
          - Recommended term: Relu, the term recommended for the deposits
          - Recommended interest_rate: Relu, the interest rate recommended for the deposits
        
        Arguments:
          user_feature_dim: int, dimension of user features
          product_feature_dim: int, dimension of product features
          cross_feature_dim: int,
          hidden_dim: int, hidden layer dimension
          embedding_dim: int, tower output dimension
        """
        super(ThreeTowerModel,self).__init__()
        # User feature tower
        self.user_tower=nn.Sequential(
            nn.Linear(user_feature_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,embedding_dim)
        )
        # Product feature tower
        self.product_tower=nn.Sequential(
            nn.Linear(product_feature_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,embedding_dim)
        )
        # Interaction feature tower
        self.cross_env_tower=nn.Sequential(
            nn.Linear(cross_feature_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,embedding_dim)
        )
        
        concat_dim=3*embedding_dim  # dimension of the concatenated result
        
        # Probability of deposits (y)
        self.branch1=nn.Sequential(
            nn.Linear(concat_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()
        )
        # Recommended term
        self.branch2=nn.Sequential(
            nn.Linear(concat_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,5),
            nn.Softmax(dim=1)
        )

        # Recommended interest rate
        self.branch3=nn.Sequential(
            nn.Linear(concat_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self,user_feat,prod_feat,cross_feat):
        """
        Forward propagation:
          Caculate the three tower and then concatenate
          Input into the three branch/result towers
          
        Arguments:
          user_feat: Tensor, (batch_size, user_feature_dim)
          prod_feat: Tensor, (batch_size, product_feature_dim)
          cross_feat: Tensor, (batch_size, cross_feature_dim)
        Return:
          out1: Tensor,(batch_size, 1),probability
          out2: Tensor, term
          out3: Tensor, interest_rate
        """
        user_emb=self.user_tower(user_feat)
        prod_emb=self.product_tower(prod_feat)
        cross_emb=self.cross_env_tower(cross_feat)
        
        combined=torch.cat([user_emb,prod_emb,cross_emb],dim=1)
        out1=self.branch1(combined)
        out2=self.branch2(combined)
        out3=self.branch3(combined)
        return out1,out2,out3
    
#####################################################
# 5. Training
#####################################################

def train_and_evaluate(model,train_loader,val_loader,criterion_y,criterion_term,criterion_ir,optimizer,num_epochs=3,alpha_y=1.0,alpha_term=1.0,alpha_ir=1.0):
    """
      loss=alpha_y*BCELoss(y_pred, y_true)+alpha_term*CELoss(term_pred, term_true)+alpha_ir*MSELoss(ir_pred,ir_true)
      Alphas are the weights to adjust. Can adjust based on importance.
      For simplicity I just pick 1 1 1
    """
    for epoch in range(num_epochs):
        model.train()
        total_train_loss=0.0
        for user_batch,prod_batch,cross_batch,y_batch,term_batch,ir_batch in train_loader:
            optimizer.zero_grad()
            pred_y,pred_term,pred_ir=model(user_batch,prod_batch,cross_batch)
            loss_y=criterion_y(pred_y,y_batch)
            loss_term=criterion_term(pred_term,term_batch.squeeze().long())
            loss_ir=criterion_ir(pred_ir,ir_batch.unsqueeze(1))
            loss=alpha_y*loss_y+alpha_term*loss_term+alpha_ir*loss_ir
            loss.backward()
            optimizer.step()
            total_train_loss+=loss.item()*user_batch.size(0)

        avg_train_loss = total_train_loss/len(train_loader.dataset)
        # evaluation
        model.eval()
        total_val_loss=0.0
        with torch.no_grad():
            for user_batch,prod_batch,cross_batch,y_batch,term_batch,ir_batch in val_loader:
                pred_y,pred_term,pred_ir=model(user_batch, prod_batch, cross_batch)
                loss_y=criterion_y(pred_y,y_batch)
                loss_term=criterion_term(pred_term,term_batch.squeeze().long())
                loss_ir=criterion_ir(pred_ir,ir_batch.unsqueeze(1))
                loss=alpha_y*loss_y+alpha_term*loss_term + alpha_ir*loss_ir
                total_val_loss+=loss.item()*user_batch.size(0)
        avg_val_loss=total_val_loss/len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print("Training complete.")
    return avg_val_loss


def evaluate_on_test(model,test_loader,criterion_y,criterion_term,criterion_ir,alpha_y=1.0,alpha_term=1.0,alpha_ir=1.0):
    """
    On testing data, for final evaluation
    """
    model.eval()
    total_test_loss=0.0
    total_correct=0
    total_samples=0
    with torch.no_grad():
        for user_b,prod_b,cross_b, y_b,term_b,ir_b in test_loader:
            py,pterm,pir=model(user_b, prod_b, cross_b)
            loss_y=criterion_y(py,y_b)
            loss_term=criterion_term(pterm, term_b)
            loss_ir=criterion_ir(pir,ir_b.unsqueeze(1))
            loss=alpha_y*loss_y +alpha_term*loss_term+alpha_ir*loss_ir
            total_test_loss+=loss.item()*user_b.size(0)

            # To give the recommended term. Need to return this result if needed
            pred_term_class=torch.argmax(pterm,dim=1)
            correct_term=(pred_term_class==term_b.squeeze().long()).sum().item()

            pred_label=(py>=0.5).float()      
            correct=(pred_label==y_b).sum().item()
            total_correct+=correct
            total_samples+=y_b.size(0)
    test_accuracy=total_correct/total_samples
    avg_test_loss=total_test_loss/len(test_loader.dataset)
    return avg_test_loss,test_accuracy

#####################################################
# 6. Predict individual customer's preferences/Recommend products and services
#####################################################


def predict_new_data(file_path):
    """

    Here we illustrate how our model could be deployed in real business situation:
    The predicted deposit probability and label could be used to roughly filter out the customers who have little chance to deposit hence there is no 
    need to do marketing and we could reduce the campaign cost on them. The recommended term and interest rate calculated from the model could be marketed
    to targeted individual customer.
    
    Argument:
        File path of the customer data is the only argument. The data file must have the columns [age, job, marital, education,	default	
        balance	housing	loan, contact, day, month, duration, campaign, pdays, previous, poutcome]. Columns[y, deposit_amount, term, 
        interest_rate] must appear but could be filled with NAs or 0 if necessary. It is strongly recommended to fill the last three columns with the initial
        default (may be estimated from Group A) recommended deposit amount, term and interest to get better model results.

    Returns:
        The model will return the customer's probility of deposit, the label(based on the probability and threshold), preferred term and interest rate.
        Note that very small recommended interest rate and 0 recommended term mean that this customer may not deposit anyway.

    """
    df=pd.read_csv(file_path)
    user_feature_cols=['age','job','marital','education','default','balance','housing','loan']
    product_feature_cols=['deposit_amount','interest_rate'] 
    cross_feature_cols=['contact','day','month','duration','campaign','pdays','previous','poutcome']
    
    user_feats=df[user_feature_cols].values
    product_feats=df[product_feature_cols].values
    cross_feats=df[cross_feature_cols].values
    user_torch=torch.tensor(user_feats,dtype=torch.float32)
    product_torch=torch.tensor(product_feats,dtype=torch.float32)
    cross_torch=torch.tensor(cross_feats,dtype=torch.float32)

    final_model.eval() 
    with torch.no_grad():
        y_pred,term_pred,ir_pred=final_model(user_torch,product_torch,cross_torch)

    deposit_prob=y_pred.squeeze().cpu().numpy()
    deposit_label=(y_pred>= 0.3).int().cpu().numpy() # Predicted as 1 threshold. Could be adjusted
    deposit_products=["current","three_months","six_months","one_year","two_year"]
    term_pred_class = torch.argmax(term_pred, dim=1).cpu().numpy()
    term_pred_label=[deposit_products[i] for i in term_pred_class] 
    interest_rate_pred=ir_pred.squeeze().cpu().numpy()

    df=pd.read_csv(file_path)
    df["deposit_probability"]=deposit_prob
    df["predicted_deposit_label"]=deposit_label
    df["recommended_term"]=term_pred_label
    df["recommended_interest_rate"]=interest_rate_pred

    return df

#####################################################
# 7. Main
#####################################################

if __name__ == "__main__":
    # 1. Data loading and preprocessing
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data_file ='../data/processed/banking_marketing_train_processed.csv'
    user_feats,product_feats,cross_feats,label_y,label_term,label_ir=load_and_preprocess_data(data_file)
    # 2. Dataset segmentation
    (X_user_train,X_user_val,X_user_test,X_product_train,X_product_val, X_product_test,X_cross_train,X_cross_val,X_cross_test,y_train,y_val,y_test,
     term_train,term_val,term_test,ir_train,ir_val,ir_test)=split_train_val_test(user_feats, product_feats, cross_feats,label_y,label_term,label_ir, 
        train_ratio=0.6,val_ratio=0.2,test_ratio=0.2,random_state=42)
    
    train_user_feats=torch.tensor(X_user_train,dtype=torch.float32)
    train_prod_feats=torch.tensor(X_product_train,dtype=torch.float32)
    train_cross_feats=torch.tensor(X_cross_train,dtype=torch.float32)
    train_y=torch.tensor(y_train.reshape(-1,1),dtype=torch.float32)
    train_term=torch.tensor(term_train,dtype=torch.long)
    train_ir=torch.tensor(ir_train,dtype=torch.float32)
    
    val_user_feats=torch.tensor(X_user_val,dtype=torch.float32)
    val_prod_feats=torch.tensor(X_product_val,dtype=torch.float32)
    val_cross_feats=torch.tensor(X_cross_val,dtype=torch.float32)
    val_y=torch.tensor(y_val.reshape(-1,1),dtype=torch.float32)
    val_term=torch.tensor(term_val,dtype=torch.long)
    val_ir=torch.tensor(ir_val,dtype=torch.float32)
    
    test_user_feats=torch.tensor(X_user_test,dtype=torch.float32)
    test_prod_feats=torch.tensor(X_product_test,dtype=torch.float32)
    test_cross_feats=torch.tensor(X_cross_test,dtype=torch.float32)
    test_y= torch.tensor(y_test.reshape(-1,1),dtype=torch.float32)
    test_term=torch.tensor(term_test,dtype=torch.long)
    test_ir=torch.tensor(ir_test,dtype=torch.float32)
    
    # 3. DataLoader
    train_dataset=torch.utils.data.TensorDataset(train_user_feats,train_prod_feats,train_cross_feats,train_y,train_term,train_ir)
    val_dataset=torch.utils.data.TensorDataset(val_user_feats,val_prod_feats,val_cross_feats,val_y,val_term,val_ir)
    test_dataset=torch.utils.data.TensorDataset(test_user_feats,test_prod_feats,test_cross_feats,test_y,test_term,test_ir)
    
    batch_size=32
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    
    # Hyperparameters
    hidden_dim_candidates=[4,8,16]
    embedding_dim_candidates=[4,8,16]
    learning_rate_candidates=[0.1,0.05,0.01,0.005,0.001]
    # hidden_dim_candidates=[16]
    # embedding_dim_candidates=[16]
    # learning_rate_candidates=[0.1]

    best_config=None
    best_val_loss=float('inf')
    # To find the best combo/config of hyperparameters
    for hd in hidden_dim_candidates:
        for ed in embedding_dim_candidates:
            for lr in learning_rate_candidates:
                # initialize model
                user_dim=train_user_feats.shape[1]
                prod_dim=train_prod_feats.shape[1]
                cross_dim=train_cross_feats.shape[1]
                model = ThreeTowerModel(user_dim,prod_dim,cross_dim,hidden_dim=hd,embedding_dim=ed)
                # Losses
                # Could test on other loss functions
                criterion_y=nn.BCELoss() 
                criterion_term=nn.CrossEntropyLoss() 
                criterion_ir=nn.MSELoss() 
                # Optimizer
                optimizer=optim.Adam(model.parameters(),lr=lr)
                # training
                val_loss = train_and_evaluate(model,train_loader,val_loader,criterion_y,criterion_term,criterion_ir,optimizer,num_epochs=3,alpha_y=1.0,alpha_term=1.0,alpha_ir=1.0)
                
                if val_loss<best_val_loss:
                    best_val_loss=val_loss
                    best_config=(hd,ed,lr)
    
    print(f"---Final Hyperparameters: hidden_dim={best_config[0]}, "
          f"embedding_dim={best_config[1]}, lr={best_config[2]}, "
          f"ValLoss={best_val_loss:.4f}")
    # Train again, using best hyperparameters
    best_hidden_dim,best_embedding_dim,best_lr=best_config
    
    final_model=ThreeTowerModel(user_dim,prod_dim,cross_dim,hidden_dim=best_hidden_dim,embedding_dim=best_embedding_dim)
    criterion_y=nn.BCELoss()
    criterion_term=nn.CrossEntropyLoss()
    criterion_ir=nn.MSELoss()
    optimizer=optim.Adam(final_model.parameters(), lr=best_lr)

    _=train_and_evaluate(final_model,train_loader,val_loader,criterion_y,criterion_term,criterion_ir,optimizer,num_epochs=5,alpha_y=1.0,alpha_term=1.0,alpha_ir=1.0)
    
    # Evaluation on test.
    test_loss,test_accuracy=evaluate_on_test(final_model,test_loader,criterion_y,criterion_term,criterion_ir,alpha_y=1.0,alpha_term=1.0,alpha_ir=1.0)
    print(f"[Test] Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

    # Save model 
    model_dir = '../models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'B1_model.pkl')
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")

    # EXAMPLE OF PREDICTION

    new_data_file='../data/processed/banking_marketing_test_processed.csv'
    df_predictions=predict_new_data(new_data_file)
    last_four_columns=df_predictions.iloc[:, -4:]
    print(last_four_columns)





