from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json
import pandas as pd

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    location: str
    sqft: float
    bath: int
    bhk: int

pipe = pickle.load(open('RidgeModel.pkl', 'rb'))

@app.post('/pred')
def pred(input_parameters: model_input):

    input_data = input_parameters.json()
    input_dict = json.loads(input_data)

    location = input_dict['location']
    sqft = input_dict['sqft']
    bath = input_dict['bath']
    bhk = input_dict['bhk']

    input_ = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input_)[0]

    return prediction
