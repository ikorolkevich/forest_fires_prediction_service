import typing

import requests
from pydantic import BaseModel, Field

from service import settings


class OneHour(BaseModel):
    one_hour: float = Field(alias='1h')


class Weather(BaseModel):
    id: float
    main: str
    description: str
    icon: str


class Hourly(BaseModel):
    dt: int
    temp: float
    feels_like: float
    pressure: float
    humidity: float
    dew_point: float
    uvi: float
    clouds: float
    visibility: float
    wind_speed: float
    wind_gust: float = None
    wind_deg: float
    pop: float
    rain: OneHour = None
    snow: OneHour = None
    weather: typing.List[Weather]


class WeatherResponse(BaseModel):
    lat: float
    lon: float
    timezone: str
    timezone_offset: int
    hourly: typing.List[Hourly]


class WeatherApi:

    app_id = settings.WEATHER_API_KEY
    weather_url = 'https://api.openweathermap.org/'

    @classmethod
    def _build(cls, query: str) -> str:
        return f'{cls.weather_url}{query}'

    @classmethod
    def get_weather_data(cls, lat: float, lon: float) -> typing.List[Hourly]:
        query = f'data/2.5/onecall?lat={lat}&lon={lon}' \
                f'&exclude=current,minutely,alerts,daily&units=metric' \
                f'&appid={cls.app_id}'
        query = cls._build(query)
        data = cls._query(query)
        response = WeatherResponse(**data)
        return response.hourly

    @classmethod
    def _query(cls, query: str) -> dict:
        response = requests.get(query)
        if response.status_code != 200:
            raise ValueError(f'Status code != 200 on query: {query}')
        return response.json()
