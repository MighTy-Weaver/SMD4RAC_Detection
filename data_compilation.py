import glob
import math
import os
from statistics import mean

import numpy as np
import pandas as pd
from tqdm import trange, tqdm


def get_average_data(date: str, hour: str, half=False):
    """
    This function returns the average data given the starting time and ending time
    :param date: A str indicating the date
    :param hour: A str indicating the hour
    :param half: Whether it's half hour or full hour
    :return: The average data for four climate categories and the time
    """
    result = []
    if len(date.split('-')) == 3:
        y, m, d = date.split('-')[0], date.split('-')[1], date.split('-')[2]
        m = '0' + m if len(m) == 1 else m
        d = '0' + d if len(d) == 1 else d
    elif len(date.split('/')) == 3:
        y, m, d = date.split('/')[0], date.split('/')[1], date.split('/')[2]
        m = '0' + m if len(m) == 1 else m
        d = '0' + d if len(d) == 1 else d
    else:
        raise Exception("Error at {}".format(date))
    date_processed = '-'.join([y, m, d])
    time = "/".join(date_processed.split('-')) + ' ' + hour + ":00" if len(hour.split(':')[0]) == 2 else "/".join(
        date_processed.split('-')) + ' 0' + hour + ":00"
    for data in [temperature, humidity, precipitation, irradiance]:
        end_index = data[data.Time == time].index.tolist()
        assert len(end_index) == 1, "Time {}'s data have more than one or not found!".format(time)
        end_index = end_index[0]
        start_index = end_index - 30 if half else end_index - 60
        if start_index < 0:
            result.append(data.loc[end_index, 'data'])
        else:
            result.append(mean([data.loc[i, 'data'] for i in range(start_index, end_index)]))
    return [result[0], result[1], result[2], result[3], time]


if __name__ == '__main__':
    irradiance = pd.read_csv(
        'data/Meteorological_Data_Hourly_and_Half_Hourly_20201230_20210630/Meteorological_Minute_20201230_20210630/Irradiance_Minute_20201230_20210630.csv')
    precipitation = pd.read_csv(
        'data/Meteorological_Data_Hourly_and_Half_Hourly_20201230_20210630/Meteorological_Minute_20201230_20210630/Precipitation_Minute_20201230_20210630.csv')
    humidity = pd.read_csv(
        'data/Meteorological_Data_Hourly_and_Half_Hourly_20201230_20210630/Meteorological_Minute_20201230_20210630/Relative_Humidity_Minute_20201230_20210630.csv')
    temperature = pd.read_csv(
        'data/Meteorological_Data_Hourly_and_Half_Hourly_20201230_20210630/Meteorological_Minute_20201230_20210630/Temperature_Minute_20201230_20210630.csv')

    half_hour = glob.glob(
        'data/electricity_data_hourly_and_half_hourly/Electricity_half_hourly_20201230-20210630/*.csv')
    hour = glob.glob('data/electricity_data_hourly_and_half_hourly/Electricity_hourly_20201230-20210630/*.csv')

    half_hour_concat = pd.concat([pd.read_csv(i, encoding='utf-16', sep='\t') for i in half_hour],
                                 ignore_index=True).rename(
        columns={'Time (date)': 'Date', 'Time (hour)': 'Hour', 'Time (day of week)': 'Weekday', 'Total (kWh)': 'Total',
                 'AC (kWh)': 'AC', 'Light (kWh)': 'Lighting', 'Socket (kWh)': 'Socket',
                 'Water Heater (kWh)': 'WaterHeater'}).drop(
        ['Location Path', 'Location Description'], axis=1).sort_values(by=['Location', 'Date', 'Hour'])
    hour_concat = pd.concat([pd.read_csv(i, encoding='utf-16', sep='\t') for i in hour], ignore_index=True).rename(
        columns={'Time (date)': 'Date', 'Time (hour)': 'Hour', 'Time (day of week)': 'Weekday', 'Total (kWh)': 'Total',
                 'AC (kWh)': 'AC', 'Light (kWh)': 'Lighting', 'Socket (kWh)': 'Socket',
                 'Water Heater (kWh)': 'WaterHeater'}).drop(
        ['Location Path', 'Location Description'], axis=1).sort_values(by=['Location', 'Date', 'Hour'])
    half_hour_concat.to_csv('.data/electricity_data_hourly_and_half_hourly/half_hour.csv', encoding='utf-8', sep=',',
                            index=False)
    hour_concat.to_csv('.data/electricity_data_hourly_and_half_hourly/hour.csv', encoding='utf-8', sep=',',
                       index=False)

    for col in ['Time', 'Temperature', 'Irradiance', 'Precipitation', 'Humidity', 'Prev_one', 'Prev_three',  # 'WIFI',
                'Prev_five', 'Prev_one_on', 'Prev_two_on', 'Next_one_on', 'Next_two_on']:
        for csv in [half_hour_concat, hour_concat]:
            csv[col] = math.nan

    if os.path.exists('./data/Average.npy'):
        total_average_data = np.load('./data/Average.npy', allow_pickle=True).item()
    else:
        total_average_data = {'half': {}, 'full': {}}
        half_hour_date = half_hour_concat['Date'].unique()
        half_hour_hour = half_hour_concat['Hour'].unique()
        full_hour_date = hour_concat['Date'].unique()
        full_hour_hour = hour_concat['Hour'].unique()
        for d in tqdm(half_hour_date):
            for h in half_hour_hour:
                total_average_data['half'][d + h] = get_average_data(d, h, True)
        for d in tqdm(full_hour_date):
            for h in full_hour_hour:
                total_average_data['full'][d + h] = get_average_data(d, h, False)
        np.save('./data/Average.npy', total_average_data)

    print(total_average_data)

    for i in trange(len(half_hour_concat), desc="Calculating half-hourly data: "):
        half_hour_concat.loc[i, 'Temperature'] = total_average_data['half'][
            half_hour_concat.loc[i, 'Date'] + half_hour_concat.loc[i, 'Hour']][0]
        half_hour_concat.loc[i, 'Humidity'] = total_average_data['half'][
            half_hour_concat.loc[i, 'Date'] + half_hour_concat.loc[i, 'Hour']][1]
        half_hour_concat.loc[i, 'Precipitation'] = total_average_data['half'][
            half_hour_concat.loc[i, 'Date'] + half_hour_concat.loc[i, 'Hour']][2]
        half_hour_concat.loc[i, 'Irradiance'] = total_average_data['half'][
            half_hour_concat.loc[i, 'Date'] + half_hour_concat.loc[i, 'Hour']][3]
        half_hour_concat.loc[i, 'Time'] = total_average_data['half'][
            half_hour_concat.loc[i, 'Date'] + half_hour_concat.loc[i, 'Hour']][4]

        half_hour_concat.loc[i, 'Prev_one_on'] = False if i < 1 else (half_hour_concat.loc[i - 1, 'AC'] > 0)
        half_hour_concat.loc[i, 'Prev_two_on'] = half_hour_concat.loc[i, 'Prev_one_on'] if i < 2 else (
                (half_hour_concat.loc[i - 1, 'AC'] > 0) and (half_hour_concat.loc[i - 2, 'AC'] > 0))
        half_hour_concat.loc[i, 'Next_one_on'] = False if i >= len(half_hour_concat) - 1 else (
                half_hour_concat.loc[i + 1, 'AC'] > 0)
        half_hour_concat.loc[i, 'Next_two_on'] = half_hour_concat.loc[i, 'Next_one_on'] if i >= len(
            half_hour_concat) - 2 else (
                (half_hour_concat.loc[i + 1, 'AC'] > 0) and (half_hour_concat.loc[i + 2, 'AC'] > 0))
        half_hour_concat.loc[i, 'Prev_one'] = half_hour_concat.loc[i - 1, 'AC'] if i >= 1 else 0
        half_hour_concat.loc[i, 'Prev_three'] = sum(
            [half_hour_concat.loc[i - k, 'AC'] for k in range(0, 3)]) if i >= 3 else sum(
            [half_hour_concat.loc[i - k, 'AC'] for k in range(0, i)])
        half_hour_concat.loc[i, 'Prev_five'] = sum(
            [half_hour_concat.loc[i - k, 'AC'] for k in range(0, 5)]) if i >= 5 else sum(
            [half_hour_concat.loc[i - k, 'AC'] for k in range(0, i)])
        print(half_hour_concat.loc[i])

    half_hour_concat.drop(['Date', 'Hour'], axis=1).to_csv('./data/half_hour_compiled.csv', index=False)

    for i in trange(len(hour_concat), desc="Calculating hourly data: "):
        hour_concat.loc[i, 'Temperature'] = \
            total_average_data['full'][hour_concat.loc[i, 'Date'] + hour_concat.loc[i, 'Hour']][0]
        hour_concat.loc[i, 'Humidity'] = \
            total_average_data['full'][hour_concat.loc[i, 'Date'] + hour_concat.loc[i, 'Hour']][1]
        hour_concat.loc[i, 'Precipitation'] = \
            total_average_data['full'][hour_concat.loc[i, 'Date'] + hour_concat.loc[i, 'Hour']][2]
        hour_concat.loc[i, 'Irradiance'] = \
            total_average_data['full'][hour_concat.loc[i, 'Date'] + hour_concat.loc[i, 'Hour']][3]
        hour_concat.loc[i, 'Time'] = \
            total_average_data['full'][hour_concat.loc[i, 'Date'] + hour_concat.loc[i, 'Hour']][4]

        hour_concat.loc[i, 'Prev_one_on'] = False if i < 1 else (hour_concat.loc[i - 1, 'AC'] > 0)
        hour_concat.loc[i, 'Prev_two_on'] = hour_concat.loc[i, 'Prev_one_on'] if i < 2 else (
                (hour_concat.loc[i - 1, 'AC'] > 0) and (hour_concat.loc[i - 2, 'AC'] > 0))
        hour_concat.loc[i, 'Next_one_on'] = False if i >= len(hour_concat) - 1 else (
                hour_concat.loc[i + 1, 'AC'] > 0)
        hour_concat.loc[i, 'Next_two_on'] = hour_concat.loc[i, 'Next_one_on'] if i >= len(
            hour_concat) - 2 else (
                (hour_concat.loc[i + 1, 'AC'] > 0) and (hour_concat.loc[i + 2, 'AC'] > 0))
        hour_concat.loc[i, 'Prev_one'] = hour_concat.loc[i - 1, 'AC'] if i >= 1 else 0
        hour_concat.loc[i, 'Prev_three'] = sum(
            [hour_concat.loc[i - k, 'AC'] for k in range(0, 3)]) if i >= 3 else sum(
            [hour_concat.loc[i - k, 'AC'] for k in range(0, i)])
        hour_concat.loc[i, 'Prev_five'] = sum(
            [hour_concat.loc[i - k, 'AC'] for k in range(0, 5)]) if i >= 5 else sum(
            [hour_concat.loc[i - k, 'AC'] for k in range(0, i)])

    hour_concat.drop(['Date', 'Hour'], axis=1).to_csv('./data/hour_compiled.csv', index=False)
