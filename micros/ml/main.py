from fastapi import FastAPI
from joblib import load

app = FastAPI()


@app.get("/")
def read_root():
    filename = 'finalized_model.sav'
    model = load('model.joblib') 
    predictions = model.predict([[0,1,0,6,0,2,0.344167,0.363625,0.805833,0.160446]])
    print(predictions[0])
    print(type(predictions))
    return {"rentals": predictions[0]}

