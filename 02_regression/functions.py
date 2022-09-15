import numpy as np
import pandas as pd 

def train_linear_regression(X,y):
    ones = np.ones(X.shape[0])

    X = np.column_stack([ones,X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    w0 = w_full[0] 
    w = w_full[1:]
    return w0, w 

def train_linear_regression_reg(X, y, alpha=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + alpha * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]

def rmse(y_preds,y):

    mse = np.sum((y_preds-y)**2)/len(y_preds)
    rmse = np.sqrt(mse)
    return rmse



def split_dataset(df,val_size,test_size,seed):
    
    idxs = df.index.values
    df_new = df.copy()

    # Seed
    np.random.seed(seed)
    np.random.shuffle(idxs)

    n = len(df_new)
    n_val = int(n*val_size)
    n_test = int(n*test_size)
    n_train = n - (n_val+n_test)

    df_train = df_new.iloc[idxs[:n_train]].reset_index(drop=True)
    df_val = df_new.iloc[idxs[n_train:n_train+n_val]].reset_index(drop=True)
    df_test = df_new.iloc[idxs[n_train+n_val:]].reset_index(drop=True)

    y_train = np.log1p(df_train['median_house_value'].values)
    y_val = np.log1p(df_val['median_house_value'].values)
    y_test = np.log1p(df_test['median_house_value'].values)

    del df_train['median_house_value']
    del df_val['median_house_value']
    del df_test['median_house_value']

    return (df_train,y_train),(df_val,y_val),(df_test,y_test)