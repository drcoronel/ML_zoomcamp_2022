import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from model_functs import * 

import pickle 

# Parameter definition

C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin' 

# Data Preparation

df = pd.read_csv('data-week-3.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
y_test = df_test.churn.values 

## Validation 

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = round(roc_auc_score(y_val, y_pred),3)
    scores.append(auc)
    print(f"Using C = {C} \n")
    print(f"Fold {fold} Validation result:{auc}")
    fold += 1

print(f"C={C} AUC {np.mean(scores)} +- { np.std(scores)} ")

## Train final model

dv, model = train(df_full_train, df_full_train.churn.values, C=C)
y_pred = predict(df_test, dv, model)

auc = round(roc_auc_score(y_test, y_pred),3)

print(f'Test auc={auc}')

print(f"Writing model to {output_file}")
with open(output_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)
print(f"Model written to {output_file}")