import datetime

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
