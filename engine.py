import pickle
from utils import make_period_time
from fastapi import FastAPI
import re
import seaborn as sns
import matplotlib.pyplot as plt
import os
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask


app = FastAPI(title='Sale Predict')

filename_model = "models/bamland_predictor.pickle"
filename_scaler = "models/bamland_scaler.pickle"

modelRFR = pickle.load(open(filename_model, "rb"))
scaler = pickle.load(open(filename_scaler, "rb"))


delimiter_pattern = r'[-.,_/]'


@app.get("/sale_predict")
async def get_strings(dates: dict):
    start_date = dates['start_time']
    start_time = re.split(delimiter_pattern, start_date)
    start_time = [int(x) for x in start_time]

    end_date = dates['end_time']
    end_time = re.split(delimiter_pattern, end_date)
    end_time = [int(x) for x in end_time]

    period_time = make_period_time(start_time, end_time)

    inputs = scaler.transform(period_time)

    predictions = modelRFR.predict(inputs)
    return {"string1": start_date, "string2": end_date, "total_sale": predictions.sum(), "predictions": list(predictions)}


def cleanup(file_path: str) -> None:
    os.remove(file_path)


@app.get("/figure")
async def figure(start_date: str, end_date: str):
    start_time = re.split(delimiter_pattern, start_date)
    start_time = [int(x) for x in start_time]

    end_time = re.split(delimiter_pattern, end_date)
    end_time = [int(x) for x in end_time]

    period_time = make_period_time(start_time, end_time)

    inputs = scaler.transform(period_time)

    predictions = modelRFR.predict(inputs)
    sns.lineplot(x=range(len(predictions)), y=predictions, label='Sales Prediction')
    plt.savefig('sale.png')
    file_path = 'sale.png'
    background_task = BackgroundTask(cleanup, file_path)
    return FileResponse(file_path, background=background_task, media_type="image/png")