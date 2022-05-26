import datetime

import pandas as pd
import pytz
from peewee import *

from service.settings import DB_PARAMS


db = PostgresqlDatabase(
    database=DB_PARAMS.dbname,
    user=DB_PARAMS.user,
    password=DB_PARAMS.password,
    host=DB_PARAMS.host,
    port=DB_PARAMS.port
)


class WeatherDataHourly(Model):
    dt = DateTimeField(unique=True)
    temp = FloatField()
    dew_point = FloatField()
    pressure = FloatField()
    humidity = FloatField()
    wind_speed = FloatField()
    rain_1h = FloatField()
    snow_1h = FloatField()

    class Meta:
        database = db


class ForestFirePredictions(Model):
    date = DateField(unique=True)
    temp = FloatField()
    dew_point = FloatField()
    wind_speed = FloatField()
    humidity = FloatField()
    pressure = FloatField()
    rainfall_24h = FloatField()
    days_from_last_rain = IntegerField()
    value = FloatField()

    class Meta:
        database = db


with db:
    for table in [WeatherDataHourly, ForestFirePredictions]:
        table.create_table()

if __name__ == '__main__':

    with db:
        # example tomorrow predict
        today = datetime.datetime.now()
        today = datetime.datetime(2021, 8, 7, 12, 0)
        # today = datetime.datetime(today.year - 1, today.month, today.day + 1, 12, 0)
        today_one_year_ago = datetime.datetime(today.year - 1, today.month, today.day, today.hour, today.minute)
        sum_rainfall_last24_h = 0
        days_wo_rain = 0
        for a in WeatherDataHourly.select().where((WeatherDataHourly.dt < today) & (WeatherDataHourly.dt > today_one_year_ago)).order_by(WeatherDataHourly.dt.desc()):
            date_time = a.dt
            print(date_time)
            if date_time.hour == 12:
                if sum_rainfall_last24_h < 2.5:
                    days_wo_rain += 1
                else:
                    break
                sum_rainfall_last24_h = 0
            sum_rainfall_last24_h += a.snow_1h + a.rain_1h
        print(days_wo_rain)

    # insert weather data
    # df = pd.read_csv(
    #     '/home/ikorolkevich/git/forest_fires_prediction_service/'
    #     'forest_fires_classifier/data_processing/normalized_ds.csv'
    #     )
    # with db:
    #     created = 0
    #     for _, row in df.iterrows():
    #         dt = datetime.datetime.fromtimestamp(row['dt'], tz=pytz.UTC)
    #         try:
    #             WeatherDataHourly.get(dt=dt)
    #         except WeatherDataHourly.DoesNotExist:
    #             rain = row['rain_1h']
    #             rain = rain if pd.isna(rain) is False else 0
    #             snow = row['snow_1h']
    #             snow = snow if pd.isna(snow) is False else 0
    #             wd = WeatherDataHourly(
    #                 dt=dt,
    #                 temp=row['temp'],
    #                 dew_point=row['dew_point'],
    #                 pressure=row['pressure'],
    #                 humidity=row['humidity'],
    #                 wind_speed=row['wind_speed'],
    #                 rain_1h=rain,
    #                 snow_1h=snow
    #             )
    #             wd.save()
    #             created += 1
    #         print(created)

    # # insert old ff probs
    # df = pd.read_csv(
    #     '/home/ikorolkevich/git/forest_fires_prediction_service/'
    #     'forest_fires_classifier/data_processing/new/prepared_last.csv'
    # )
    # with db:
    #     created = 0
    #     for _, row in df.iterrows():
    #         label = row['labels']
    #         if label == 1:
    #             label = 10
    #         elif label == 2:
    #             label = 35
    #         elif label == 3:
    #             label = 85
    #         elif label == 0:
    #             label = 0
    #         else:
    #             label = 95
    #
    #         dt = datetime.datetime.fromisoformat(row['dt']).date()
    #         try:
    #             ForestFirePredictions.get(ForestFirePredictions.date==dt)
    #         except ForestFirePredictions.DoesNotExist:
    #             ffp = ForestFirePredictions(
    #                 date=dt,
    #                 temp=row['temp'],
    #                 dew_point=row['dew_point'],
    #                 wind_speed=row['wind_speed'],
    #                 humidity=row['humidity'],
    #                 pressure=row['pressure'],
    #                 rainfall_24h=row['rainfall_last24_h'],
    #                 days_from_last_rain=row['days_from_last_rain'],
    #                 value=label,
    #             )
    #             ffp.save()
    #             created += 1
    #         print(created)
