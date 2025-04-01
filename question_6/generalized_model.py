###################################################
#   1. Packages
###################################################
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

###################################################
#   2. Data Preprocessing and Feature Engineering
###################################################
# Data preprocessing and feature engineering

def load_and_preprocess_data(file_path):
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

    # transform interest_rate,deposut_amount,term and binarize response variable y
    if 'interest_rate' in df.columns:
        df['interest_rate']=df['interest_rate'].fillna(0).astype(float)
    if 'deposit_amount' in df.columns:
        df['deposit_amount']=df['deposit_amount'].fillna(0).astype(float)
    if 'term' in df.columns:
        df['term']=df['term'].fillna(0)
    if 'y' in df.columns:
        df['y']=df['y'].map({'yes':1,'no':0})
        df['y']=df['y'].astype(float)
    
    # Encode categorical variables. May use other encoding methods, for example for education. Remove age
    feature_encoded={'job','marital','education','default','housing','loan','contact','day','month','campaign','pdays','previous','poutcome',"term"}
    le=LabelEncoder()
    for feature in feature_encoded:
        df[feature]=le.fit_transform(df[feature])

    # Labels/Outcomes
    label_y=df['y'].values
    label_term=df['term'].values
    label_ir=df['interest_rate'].values
    
    # Categorize the features (could be adjusted)
    user_feature_cols=['age','job','marital','education','default','balance','housing','loan']
    product_feature_cols=['deposit_amount','interest_rate'] 
    cross_feature_cols=['contact','day','month','duration','campaign','pdays','previous','poutcome']
    
    user_continuous=['age','balance']
    product_continuous=['deposit_amount','interest_rate']
    cross_continuous=['duration']

    scaler_user=StandardScaler()
    df[user_continuous]=scaler_user.fit_transform(df[user_continuous])
    
    scaler_product=StandardScaler()
    df[product_continuous]=scaler_product.fit_transform(df[product_continuous])
    
    scaler_cross=StandardScaler()
    df[cross_continuous]=scaler_cross.fit_transform(df[cross_continuous])
    
    user_cat=[col for col in user_feature_cols if col not in user_continuous]
    product_cat=[col for col in product_feature_cols if col not in product_continuous]
    cross_cat=[col for col in cross_feature_cols if col not in cross_continuous]
    
    user_features=np.concatenate([df[user_continuous].values,df[user_cat].values],axis=1)
    product_features=np.concatenate([df[product_continuous].values,df[product_cat].values],axis=1)
    cross_features=np.concatenate([df[cross_continuous].values,df[cross_cat].values],axis=1)

    return  user_features, product_features,cross_features,label_y,label_term,label_ir
    
###################################################
#   3. Dataset Segmentation
###################################################
def split_train_val_test(user_feats, product_feats,cross_feats,label_y,label_term,label_ir,train_ratio=0.6,val_ratio=0.2,test_ratio=0.2,random_state=42):
    """
    Split the dataset into training, validation and testing set.
    """
    # need to seperate twice since the default method only support binay split
    X_user_train,X_user_temp,X_product_train, X_product_temp,X_cross_train,X_cross_temp,y_train,y_temp,term_train,term_temp,ir_train,ir_temp=train_test_split(
        user_feats,product_feats,cross_feats,label_y, label_term, label_ir,test_size=(1-train_ratio), random_state=random_state)
    
    val_portion=val_ratio/(val_ratio+test_ratio)

    X_user_val,X_user_test,X_product_val,X_product_test,X_cross_val, X_cross_test,y_val, y_test,term_val, term_test,ir_val, ir_test=train_test_split(
        X_user_temp,X_product_temp,X_cross_temp,y_temp,term_temp,ir_temp,test_size=(1-val_portion),random_state=random_state)
    
    return (X_user_train,X_user_val,X_user_test,X_product_train,X_product_val,X_product_test,X_cross_train,X_cross_val,X_cross_test,y_train,y_val,y_test,
            term_train,term_val,term_test,ir_train,ir_val,ir_test)

###################################################
#   4. Modified Deep Structured Semantic Model (Three Towers)
###################################################
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

###################################################
#   5. Training
###################################################
def train_and_evaluate(model,train_loader,val_loader,criterion_y,criterion_term,criterion_ir,optimizer,num_epochs=5,alpha_y=1.0,alpha_term=1.0,alpha_ir=1.0):
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

###################################################
#   6. Main
###################################################

"""

A windoe is used to determine the paths of training set and prediction set.
The results will be directly printed in the window also. Note that only the predicted results will be displayed

"""
if __name__ == "__main__":
    def predict_new_data(file_path):
        user_feats,product_feats,cross_feats, _, _, _ = load_and_preprocess_data(file_path)
        user_torch=torch.tensor(user_feats,dtype=torch.float32)
        product_torch=torch.tensor(product_feats,dtype=torch.float32)
        cross_torch=torch.tensor(cross_feats,dtype=torch.float32)
        final_model.eval() 
        with torch.no_grad():
            y_pred,term_pred,ir_pred=final_model(user_torch,product_torch,cross_torch)
        deposit_prob=y_pred.squeeze().cpu().numpy()
        deposit_label=(y_pred>=0.3).int().cpu().numpy()  # the threshold could be changed
        deposit_products=["current","three_months","six_months","one_year","two_year"]
        term_pred_class=torch.argmax(term_pred, dim=1).cpu().numpy()
        term_pred_label=[deposit_products[i] for i in term_pred_class]
        interest_rate_pred=ir_pred.squeeze().cpu().numpy()
        df=pd.read_csv(file_path)
        df["deposit_probability"]=deposit_prob
        df["predicted_deposit_label"]=deposit_label
        df["recommended_term"]=term_pred_label
        df["recommended_interest_rate"]=interest_rate_pred
        return df

    root=tk.Tk()
    root.title("Recommendation")
    root.geometry("800x800")

    train_path=None
    predict_path=None

    def get_train_path():
        global train_path
        train_path=filedialog.askopenfilename(title="Choose training data path(compulsory)")
        train_entry.delete(0,tk.END)
        train_entry.insert(0,train_path)

    def get_predict_path():
        global predict_path
        predict_path=filedialog.askopenfilename(title="Choose prediction data path(compulsory)")
        predict_entry.delete(0,tk.END)
        predict_entry.insert(0,predict_path)

    def confirm():
        if not predict_path:
            print("WRONG")
            return
        if not train_path:
            print("WRONG")
            return
        print(f"training path: {train_path}")
        print(f"prediction path: {predict_path}")
        status_label.config(text="Model training...")
        root.update_idletasks()
        
        data_file = train_path
        user_feats, product_feats, cross_feats, label_y, label_term, label_ir=load_and_preprocess_data(data_file)
        (X_user_train, X_user_val, X_user_test,X_product_train,X_product_val,X_product_test,X_cross_train,X_cross_val,X_cross_test,y_train,y_val,y_test,
         term_train, term_val,term_test,ir_train,ir_val,ir_test)=split_train_val_test(user_feats,product_feats,cross_feats,label_y,label_term,label_ir, 
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
        test_y=torch.tensor(y_test.reshape(-1,1),dtype=torch.float32)
        test_term=torch.tensor(term_test,dtype=torch.long)
        test_ir=torch.tensor(ir_test,dtype=torch.float32)
        
        batch_size=32
        train_dataset=torch.utils.data.TensorDataset(train_user_feats,train_prod_feats,train_cross_feats,train_y,train_term,train_ir)
        val_dataset=torch.utils.data.TensorDataset(val_user_feats,val_prod_feats,val_cross_feats,val_y,val_term,val_ir)
        test_dataset=torch.utils.data.TensorDataset(test_user_feats,test_prod_feats,test_cross_feats,test_y,test_term,test_ir)
        
        train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
        test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size, shuffle=False)
        
        hidden_dim=16
        embedding_dim=4
        learning_rate=0.001
        user_dim=train_user_feats.shape[1]
        prod_dim=train_prod_feats.shape[1]
        cross_dim=train_cross_feats.shape[1]
        
        global final_model
        final_model=ThreeTowerModel(user_dim, prod_dim, cross_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
        criterion_y=nn.BCELoss()
        criterion_term=nn.CrossEntropyLoss()
        criterion_ir=nn.MSELoss()
        optimizer=optim.Adam(final_model.parameters(), lr=learning_rate)
        
        _=train_and_evaluate(final_model,train_loader,val_loader,criterion_y,criterion_term,criterion_ir,optimizer,num_epochs=5,alpha_y=1.0,alpha_term=1.0,alpha_ir=1.0)
        df_predictions=predict_new_data(predict_path)
        result_text=str(df_predictions.iloc[:, -4:])
        status_label.config(text=result_text)

    tk.Label(root,text="Choose training data path(optional):").pack()
    train_entry=tk.Entry(root,width=50)
    train_entry.pack()
    tk.Button(root,text="browse...", command=get_train_path).pack()
    
    tk.Label(root,text="Choose prediction data path(compulsory):").pack()
    predict_entry=tk.Entry(root, width=50)
    predict_entry.pack()
    tk.Button(root,text="browse...",command=get_predict_path).pack()
    
    tk.Button(root,text="confirm",command=confirm).pack()
    status_label = tk.Label(root, text="", justify="left")
    status_label.pack()
    root.mainloop()




