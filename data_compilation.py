import glob
import math
from statistics import mean

import pandas as pd
from tqdm import tqdm, trange


def get_average_data(date: str, hour: str, half=False):
    """
    This function returns the average data given the starting time and ending time
    :param date: A str indicating the date
    :param hour: A str indicating the hour
    :param half: Whether it's half hour or full hour
    :return: The average data for four climate categories
    """
    result = []
    time = "/".join(date.split('-')) + ' ' + hour + ":00" if len(hour.split(':')[0]) == 2 else "/".join(
        date.split('-')) + ' 0' + hour + ":00"
    for data in [temperature, humidity, precipitation, irradiance]:
        end_index = data[data.Time == time].index.tolist()
        assert len(end_index) == 1, "Time {}'s data have more than one or not found!".format(time)
        end_index = end_index[0]
        start_index = end_index - 30 if half else end_index - 60
        if start_index < 0:
            result.append(data.loc[end_index, 'data'])
        else:
            result.append(mean([data.loc[i, 'data'] for i in range(start_index, end_index)]))
    return result[0], result[1], result[2], result[3]


if __name__ == '__main__':
    irradiance = pd.read_csv(
        './Meteorological_Data_Hourly_and_Half_Hourly_20201230_20210630/Meteorological_Minute_20201230_20210630/Irradiance_Minute_20201230_20210630.csv')
    precipitation = pd.read_csv(
        './Meteorological_Data_Hourly_and_Half_Hourly_20201230_20210630/Meteorological_Minute_20201230_20210630/Precipitation_Minute_20201230_20210630.csv')
    humidity = pd.read_csv(
        './Meteorological_Data_Hourly_and_Half_Hourly_20201230_20210630/Meteorological_Minute_20201230_20210630/Relative_Humidity_Minute_20201230_20210630.csv')
    temperature = pd.read_csv(
        './Meteorological_Data_Hourly_and_Half_Hourly_20201230_20210630/Meteorological_Minute_20201230_20210630/Temperature_Minute_20201230_20210630.csv')

    half_hour = glob.glob('./electricity_data_hourly_and_half_hourly/Electricity_half_hourly_20201230-20210630/*.csv')
    hour = glob.glob('./electricity_data_hourly_and_half_hourly/Electricity_hourly_20201230-20210630/*.csv')

    half_hour_concat = pd.concat([pd.read_csv(i, encoding='utf-16', sep='\t') for i in half_hour],
                                 ignore_index=True).rename(
        columns={'Time (date)': 'Date', 'Time (hour)': 'Hour', 'Time (day of week)': 'Weekday', 'Total (kWh)': 'Total',
                 'AC (kWh)': 'AC', 'Light (kWh)': 'Lighting', 'Socket (kWh)': 'Socket',
                 'Water Heater (kWh)': 'WaterHeater'}).drop(
        ['Location Path', 'Location Description'], axis=1)
    hour_concat = pd.concat([pd.read_csv(i, encoding='utf-16', sep='\t') for i in hour], ignore_index=True).rename(
        columns={'Time (date)': 'Date', 'Time (hour)': 'Hour', 'Time (day of week)': 'Weekday', 'Total (kWh)': 'Total',
                 'AC (kWh)': 'AC', 'Light (kWh)': 'Lighting', 'Socket (kWh)': 'Socket',
                 'Water Heater (kWh)': 'WaterHeater'}).drop(
        ['Location Path', 'Location Description'], axis=1)
    half_hour_concat.to_csv('./electricity_data_hourly_and_half_hourly/half_hour.csv', encoding='utf-8', sep=',',
                            index=False)
    hour_concat.to_csv('./electricity_data_hourly_and_half_hourly/hour.csv', encoding='utf-8', sep=',',
                       index=False)

    for col in ['Temperature', 'Irradiance', 'Precipitation', 'Humidity', 'WIFI', 'Prev_one', 'Prev_three', 'Prev_five',
                'Prev_one_on', 'Prev_two_on']:
        for csv in [half_hour_concat, hour_concat]:
            csv[col] = math.nan

    for ind, csv in enumerate([half_hour_concat, hour_concat]):
        for date in tqdm(csv['Date'].unique()):
            for hour in csv[csv.Date == date]['Hour'].unique():
                t, h, p, i = get_average_data(date, hour, ind == 0)
                csv[(csv.Date == date) & (csv.Hour == hour)]['Temperature'] = t
                csv[(csv.Date == date) & (csv.Hour == hour)]['Humidity'] = h
                csv[(csv.Date == date) & (csv.Hour == hour)]['Irradiance'] = i
                csv[(csv.Date == date) & (csv.Hour == hour)]['Precipitation'] = p

    half_hour_concat.to_csv('./half_hour_compiled_without_prev.csv', index=False, encoding='utf-8', sep=',')
    hour_concat.to_csv('./hour_compiled_without_prev.csv', index=False, encoding='utf-8', sep=',')

    for csv in [half_hour_concat, hour_concat]:
        for i in trange(len(csv)):
            csv.loc[i, 'Prev_one_on'] = False if i < 1 else (csv.loc[i - 1, 'AC'] > 0)
            csv.loc[i, 'Prev_two_on'] = csv.loc[i, 'Prev_one_on'] if i < 2 else (
                    (csv.loc[i - 1, 'AC'] > 0) and (csv.loc[i - 2, 'AC'] > 0))
            csv.loc[i, 'Prev_one'] = csv.loc[i - 1, 'AC'] if i >= 1 else 0
            csv.loc[i, 'Prev_three'] = sum([csv.loc[i - k, 'AC'] for k in range(i - 3, i)]) if i >= 3 else sum(
                [csv.loc[i - k, 'AC'] for k in range(0, i)])
            csv.loc[i, 'Prev_five'] = sum([csv.loc[i - k, 'AC'] for k in range(i - 5, i)]) if i >= 5 else sum(
                [csv.loc[i - k, 'AC'] for k in range(0, i)])

    half_hour_concat.to_csv('./half_hour_compiled.csv', index=False, encoding='utf-8', sep=',')
    hour_concat.to_csv('./hour_compiled.csv', index=False, encoding='utf-8', sep=',')
