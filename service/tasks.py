import datetime

import dramatiq
import logging

import numpy as np
import pytz
import torch.cuda

from forest_fires_classifier.classifier import ForestFireClassifier
from service.db_models import WeatherDataHourly, db, ForestFirePredictions
from service.decorators import done_for
from service.dramatiqconfig import TASK_PARAMS
from service.settings import MODEL_PATH
from service.weather_api import WeatherApi

logger = logging.getLogger(__name__)
MODEL_KEY = 'MODEL'
UNITS = {}


@dramatiq.actor(
    max_retries=TASK_PARAMS.max_retries,
    queue_name=TASK_PARAMS.get_weather_data_queue
)
@done_for
def get_weather_data() -> None:
    data = WeatherApi.get_weather_data(61.325919, 100.493756)
    with db:
        for hour_data in data:
            weather_data = WeatherDataHourly(
                    dt=datetime.datetime.fromtimestamp(
                        hour_data.dt, tz=pytz.UTC
                    ),
                    temp=hour_data.temp,
                    dew_point=hour_data.dew_point,
                    pressure=hour_data.pressure / (1 / 0.75),
                    humidity=hour_data.humidity / 100,
                    wind_speed=hour_data.wind_speed,
                    rain_1h=hour_data.rain,
                    snow_1h=hour_data.snow
                )
            weather_data.get_or_create()


@dramatiq.actor(
    max_retries=TASK_PARAMS.max_retries,
    queue_name=TASK_PARAMS.predict_forest_fire_queue
)
@done_for
def predict_forest_fire():
    if UNITS.get(MODEL_KEY) is None:
        cl = ForestFireClassifier.load_from_checkpoint(
            MODEL_PATH
        )
        cl.to('cuda' if torch.cuda.is_available() else 'cpu')
        cl.eval()
        UNITS[MODEL_KEY] = cl

    today = datetime.datetime.today()
    # TODO remove after deploying
    today = datetime.datetime(
        today.year-1, today.month, today.day, 12, 0
    )
    with db:
        for i in range(1, 5):
            next_day = datetime.datetime(
                today.year, today.month,
                today.day + i, today.hour,
                today.minute
            )
            next_day_one_year_ago = datetime.datetime(
                next_day.year - 1, next_day.month,
                next_day.day, next_day.hour,
                next_day.minute
            )
            try:
                mid_day = WeatherDataHourly.get(dt=next_day)
            except WeatherDataHourly.DoesNotExist:
                break

            history = []
            sum_rainfall_last24_h = mid_day.snow_1h + mid_day.rain_1h
            days_wo_rain = 0

            for row in WeatherDataHourly.select().where(
                    (WeatherDataHourly.dt < next_day) &
                    (WeatherDataHourly.dt > next_day_one_year_ago)
            ).order_by(WeatherDataHourly.dt.desc()):
                if row.dt.hour == 12:
                    history.append(sum_rainfall_last24_h)
                    if sum_rainfall_last24_h < 2.5:
                        days_wo_rain += 1
                    else:
                        break
                    sum_rainfall_last24_h = 0
                sum_rainfall_last24_h += row.snow_1h + row.rain_1h

            value = UNITS[MODEL_KEY].predict(
                np.array(
                    [
                        mid_day.temp, mid_day.dew_point, mid_day.wind_speed,
                        mid_day.pressure, mid_day.humidity, days_wo_rain
                    ]
                )
            )
            weather_data = ForestFirePredictions(
                date=next_day.date(),
                temp=mid_day.temp,
                dew_point=mid_day.dew_point,
                wind_speed=mid_day.wind_speed,
                humidity=mid_day.humidity,
                pressure=mid_day.pressure,
                rainfall_24h=history[0],
                days_from_last_rain=days_wo_rain,
                value=value
            )
            weather_data.get_or_create()
