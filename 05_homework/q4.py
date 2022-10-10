import pickle
from flask import Flask, json 
from flask import request, jsonify 

app = Flask("credit-scoring")

with open('model1.bin','rb') as f:
    model = pickle.load(f)
with open('dv.bin','rb') as f:
    dv = pickle.load(f)



@app.route('/predict',methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    y_pred = float(round(y_pred,3))
    result = {
        'Approval_credit_probability' : y_pred
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)