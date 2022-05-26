import datetime

import numpy as np
import pandas as pd
import pytz
from sklearn.model_selection import train_test_split


def get_label(kp: float) -> int:
    if kp < 300:
        return 1
    if kp < 1000:
        return 2
    if kp < 4000:
        return 3
    if kp < 10000:
        return 4
    return 5


def calculate_kp(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    labels = []
    for ind, row in df.iterrows():
        kp = 0
        for i in range(1, row['days_from_last_rain']+1):
            tmp_ind = ind - i
            if tmp_ind < 0:
                break
            else:
                tmp_row = df.iloc[tmp_ind]
                kp += tmp_row['temp']*(tmp_row['temp']-tmp_row['dew_point'])
        kp += row['temp']*(row['temp']-row['dew_point'])
        label = get_label(kp)
        if label == 5:
            print(row['dt'])
        result.append(kp)
        labels.append(label)
    df['kp'] = result
    df['labels'] = labels
    return df


def calculate_sum_rainfall_last24_h(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    sum_rainfall_last24_h = 0
    for ind, row in df.iterrows():
        date_time = datetime.datetime.fromtimestamp(row['dt'], tz=pytz.UTC)
        if date_time.hour == 12:
            result.append(
                (
                    date_time, row['temp'], row['dew_point'], row['wind_speed'],
                    row['humidity'], row['pressure'], sum_rainfall_last24_h
                )
            )
            sum_rainfall_last24_h = 0
        sum_rainfall_last24_h += sum(
            list(
                filter(
                    lambda x: pd.isna(x) is False,
                    [row['rain_1h'], row['snow_1h']]
                )
            )
        )
    return pd.DataFrame(
        result,
        columns=[
            'dt', 'temp', 'dew_point', 'wind_speed', 'humidity',
            'pressure', 'rainfall_last24_h'
        ]
    )


def calculate_days_from_last_rain(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    days_from_last_rain = 0
    for ind, row in df.iterrows():
        days_from_last_rain = days_from_last_rain + 1 \
            if row['rainfall_last24_h'] < 2.5 else 0
        result.append(days_from_last_rain)
    df['days_from_last_rain'] = result
    return df


def prepare_owm_csv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[
        'dt_iso',
        'timezone',
        'city_name',
        'lat',
        'lon',
        'visibility',
        'feels_like',
        'temp_min',
        'temp_max',
        'sea_level',
        'grnd_level',
        'wind_deg',
        'wind_gust',
        'rain_3h',
        'snow_3h',
        'clouds_all',
        'weather_id',
        'weather_main',
        'weather_description',
        'weather_icon'
    ])
    df['pressure'] = df['pressure'] / (1 / 0.75)
    df['humidity'] = df['humidity'] / 100
    return df


def prepare(path_to_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_to_csv)
    df = prepare_owm_csv(df)
    df = calculate_sum_rainfall_last24_h(df)
    df = calculate_days_from_last_rain(df)
    df = calculate_kp(df)
    return df


if __name__ == '__main__':
    ds = prepare('/home/ikorolkevich/git/forest_fires_prediction_service/forest_fires_classifier/data_processing/147a76e35896a5f97b90aa1f454f1e96.csv')
    ds.to_csv('new/prepared_last.csv', index=False)
    # x = ds[
    #     [
    #         'temp', 'dew_point', 'wind_speed',
    #         'pressure', 'humidity', 'days_from_last_rain'
    #     ]
    # ].values
    # y = ds['labels'].values
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.2, stratify=y, shuffle=True
    # )
    # y_test = y_test.reshape(-1, 1)
    # y_train = y_train.reshape(-1, 1)
    # columns = [
    #     'temp', 'dew_point', 'wind_speed', 'pressure', 'humidity',
    #     'days_from_last_rain', 'label'
    # ]
    # df_train = pd.DataFrame(
    #     data=np.concatenate((x_train, y_train), axis=1),
    #     columns=columns
    # )
    # df_test = pd.DataFrame(
    #     data=np.concatenate((x_test, y_test), axis=1),
    #     columns=columns
    # )
    # df_train['label'] = df_train['label'].apply(lambda x: int(x - 1))
    # df_test['label'] = df_test['label'].apply(lambda x: int(x - 1))
    # df_train.to_csv('new/train_l.csv', index=False)
    # df_test.to_csv('new/test_l.csv', index=False)
