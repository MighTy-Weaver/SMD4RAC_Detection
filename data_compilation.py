import glob

import pandas as pd

half_hour = glob.glob('./electricity_data_hourly_and_half_hourly/Electricity_half_hourly_20201230-20210630/*.csv')
hour = glob.glob('./electricity_data_hourly_and_half_hourly/Electricity_hourly_20201230-20210630/*.csv')
full_half_hour = pd.DataFrame(
    columns=['date', 'hour', 'day', 'location', 'total', 'AC', 'lighting', 'socket', 'temperature', 'humidity',
             'irradiance', 'precipitation', 'WIFI'])
full_hour = pd.DataFrame(
    columns=['date', 'hour', 'day', 'location', 'total', 'AC', 'lighting', 'socket', 'temperature', 'humidity',
             'irradiance', 'precipitation', 'WIFI'])

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
print(list(half_hour_concat), half_hour_concat, hour_concat)

