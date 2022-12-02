from fastapi import FastAPI
from fastapi import status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from joblib import load


class Features(BaseModel):
    season: int
    mnth: int
    holiday: int
    weekday: int
    workingday: int
    weathersit: int
    temp: float
    atemp: float
    hum: float
    windspeed: float

    class Config:
        schema_extra = {
            "example": {
                "season": 1,
                "mnth": 1,
                "holiday": 0,
                "weekday": 6,
                "workingday": 0,
                "weathersit": 2,
                "temp": 0.344167,
                "atemp": 0.363625,
                "hum": 0.805833,
                "windspeed": 0.160446
            }
        }

class Label(BaseModel):
    rentals: float

class Message(BaseModel):
    message: float

description = """# RENTALS API

## Rentals prediction

1. **season**: A numerically encoded value indicating the season (1:winter, 2:spring, 3:summer, 4:fall)
2. **mnth**: The calendar month in which the observation was made (1:January ... 12:December)
3. **holiday**: A binary value indicating whether or not the observation was made on a public holiday)
4. **weekday**: The day of the week on which the observation was made (0:Sunday ... 6:Saturday)
5. **workingday**: A binary value indicating whether or not the day is a working day (not a weekend or holiday)
6. **weathersit**: A categorical value indicating the weather situation (1:clear, 2:mist/cloud, 3:light rain/snow, 4:heavy rain/hail/snow/fog)
7. **temp**: The temperature in celsius (normalized)
8. **atemp**: The apparent ("feels-like") temperature in celsius (normalized)
9. **hum**: The humidity level (normalized)
10. **windspeed**: The windspeed (normalized)

"""

app = FastAPI(
    title="Rentals API REST",
    description=description,
    version="1.0.0",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "Salvador HM",
        "url": "http://github.com/salvadorhm",
        "email": "salvadorhm@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


@app.post(
    "/rentals/",
    response_model=Label,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Rentals prediction",
    description="Rentals prediction",
    tags=["Rentals"]
)
async def get_rentals(features:Features):
    try:
        model = load('model.joblib')
        data = [
            features.season,
            features.mnth,
            features.holiday,
            features.weekday,
            features.workingday,
            features.weathersit,
            features.temp,
            features.atemp,
            features.hum,
            features.windspeed
        ]
        predictions = model.predict([data])
        response = {"rentals": predictions[0]}
        return response
    except Exception as e:
        response = JSONResponse(
                    status_code=400,
                    content={"message":f"{e.args}"},
                )
        return response

