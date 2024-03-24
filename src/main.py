from fastapi import FastAPI
from pydantic import BaseModel
from ml.model import inference as model_inference
from ml.model import load_model_attributes as load_model
import pandas as pd
import numpy as np

app = FastAPI(
    title="API for salary prediction",
    description="An API to predict whether income\
          exceeds $50K/yr based on census data.",
    version="1.0.0",
)


class Data(BaseModel):
    model_path: str
    query_example: dict


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to our API!"}


@app.post("/inference")
async def inference(data: Data):
    model_path = data.model_path
    model = load_model(model_path)
    query = data.query_example
    dataset = pd.DataFrame(query)
    preds = model_inference(model, dataset)
    raw_labels = np.array(["<=50K", ">50K"], dtype=str)
    num_labels = model['lb'].transform(raw_labels)
    map_labels = {num_labels[0][0]: "<=50K", num_labels[1][0]: ">50K"}
    print(map_labels)
    outputs = [map_labels[pred] for pred in preds]
    return outputs
