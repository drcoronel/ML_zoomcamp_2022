import bentoml 

from bentoml.io import JSON

from pydantic import BaseModel,StrictStr


class CreditApplication(BaseModel):
    seniority: int  
    home: StrictStr 
    time: int 
    age: int 
    marital: StrictStr
    records: StrictStr 
    job: StrictStr 
    expenses: int
    income: float
    assets: float
    debt: float 
    amount: int
    price: int

model_ref = bentoml.xgboost.get("credit_risk_model:ynkqxzsptcd5lhl2") 
dv = model_ref.custom_objects['DictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("credit_risk_classifier",runners=[model_runner]) 

@svc.api(input=JSON(pydantic_model=CreditApplication),output=JSON())
async def classify(credit_application):
    application_data = credit_application.dict()
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    result = prediction[0]

    print('---------------')
    print(result)
    print('---------------')
    if result > 0.5:
        return {"status":"DECLINED"}
    
    elif result > 0.2:
        return {"status":"MAYBE"} 
    else: 
        return {"status":"APPROVE"}

