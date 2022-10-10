import pickle

with open('model1.bin','rb') as f:
    model = pickle.load(f)
with open('dv.bin','rb') as f:
    dv = pickle.load(f)


customer = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

X = dv.transform([customer])

y_pred = model.predict_proba(X)[:,1]

print(f"Customer prob to get a credit card is {y_pred}")
