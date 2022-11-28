import requests 

#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url = 'https://1zzwj7x9r6.execute-api.eu-west-1.amazonaws.com/test/predict'

data = {'url': 'http://bit.ly/mlbookcamp-pants'}

result = requests.post(url, json=data).json()
print(result)