import requests

host= "churn-serving-env.eba-sb3wme44.eu-north-1.elasticbeanstalk.com"
url = f"http://{host}/predict"

customer_id = "xyz-123"
customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}

response = requests.post(url,json=customer).json()

print(response)

if response['churn'] == True:
    print(f"sending promo email to customer {customer_id}")
else:
    print(f"Not sending promo emial to {customer_id}")
