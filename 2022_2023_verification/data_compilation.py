import glob
import os.path

import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    if not os.path.exists('./total_meteorological.csv'):
        irradiance = pd.read_csv(
            '../data/2022_2023_verification/meteorological/A_IR_223328_1142670.csv', index_col=None)
        precipitation = pd.read_csv(
            '../data/2022_2023_verification/meteorological/A_PRECIP_223328_1142670.csv', index_col=None)
        humidity = pd.read_csv(
            '../data/2022_2023_verification/meteorological/A_RH_VT_223328_1142670.csv', index_col=None)
        temperature = pd.read_csv(
            '../data/2022_2023_verification/meteorological/A_TEMP_223328_1142670.csv', index_col=None)

        # join four dataframes according to their Time column
        df = pd.merge(irradiance, precipitation, on='Time')
        df = pd.merge(df, humidity, on='Time')
        df = pd.merge(df, temperature, on='Time')
        df = df.rename({'w/m2': 'irradiance', 'mm': 'precipitation', '%': 'humidity', 'Degree Celsius': 'temperature'},
                       axis=1)
        df = df[['Time', 'irradiance', 'precipitation', 'humidity', 'temperature']]
        df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')
        df.to_csv('./meteorological.csv', index=False)

        # copy another df, for each entry in Time column, add thirty minutes, for each entry in precipitation column, divide by 2, other columns stay the same
        df_half_hour = df.copy()
        df_half_hour['Time'] = df_half_hour['Time'] + pd.Timedelta(minutes=30)
        df_half_hour['precipitation'] = df_half_hour['precipitation'] / 2
        df['precipitation'] = df['precipitation'] / 2
        total_metereological = pd.concat([df, df_half_hour], ignore_index=True)
        total_metereological = total_metereological.sort_values(by=['Time']).reset_index(drop=True)
        total_metereological.to_csv('./total_meteorological.csv', index=False)
    else:
        total_metereological = pd.read_csv('./total_meteorological.csv', index_col=None)

    if not os.path.exists('./half_hour_concat_ac.csv'):
        half_hour = glob.glob('../data/2022_2023_verification/electricity_data/*.xlsx')
        half_hour_concat = pd.concat(
            [pd.read_excel(i, index_col=None) for i in tqdm(half_hour, desc="Loading data") if
             'appliance' in list(pd.read_excel(i, index_col=None))], ignore_index=True)
        half_hour_concat_ac = half_hour_concat[half_hour_concat['appliance'] == 'AC'].reset_index(drop=True).drop(
            ['location_id', 'appliance', 'position'], axis=1).sort_values(by=['datetime', 'name']).reset_index(
            drop=True)
        # transfer the datetime column to the format of the meteorological data
        half_hour_concat_ac['datetime'] = pd.to_datetime(half_hour_concat_ac['datetime'], format='%Y/%m/%d  %H:%M:%S')
        half_hour_concat_ac = half_hour_concat_ac.rename({'datetime': 'Time', 'name': 'Location', 'value_kwh': 'AC'},
                                                         axis=1)

        half_hour_concat_ac.to_csv('./half_hour_concat_ac.csv', index=False)
    else:
        half_hour_concat_ac = pd.read_csv('./half_hour_concat_ac.csv', index_col=None)

    # join the two dataframes according to their Time column
    total = pd.merge(total_metereological, half_hour_concat_ac, on='Time')
    print(total)
    print(list(total))
    total = total.sort_values(by=['Time', 'Location']).reset_index(drop=True)
    total.to_csv('./total.csv', index=False)

    total_csv_list = []
    for l in total['Location'].unique():
        total_csv_room = total[total['Location'] == l].sort_values(by=['Time']).reset_index(drop=True)
        total_csv_room['Prev_one'] = total_csv_room['AC'].shift(1)
        total_csv_room['Prev_two'] = total_csv_room['AC'].shift(2)
        total_csv_room['Prev_three'] = total_csv_room['AC'].rolling(3).sum().shift(1)
        total_csv_room['Prev_five'] = total_csv_room['AC'].rolling(5).sum().shift(1)
        total_csv_room['Next_one'] = total_csv_room['AC'].shift(-1)
        total_csv_room['Next_two'] = total_csv_room['AC'].shift(-2)
        total_csv_room['Prev_one_on'] = total_csv_room['Prev_one'].apply(lambda x: 1 if x > 0 else 0)
        total_csv_room['Prev_two_on'] = total_csv_room['Prev_two'].apply(lambda x: 1 if x > 0 else 0)
        total_csv_room['Next_one_on'] = total_csv_room['Next_one'].apply(lambda x: 1 if x > 0 else 0)
        total_csv_room['Next_two_on'] = total_csv_room['Next_two'].apply(lambda x: 1 if x > 0 else 0)
        total_csv_room.drop(['Prev_two', 'Next_one', 'Next_two'], axis=1, inplace=True)
        total_csv_list.append(total_csv_room)
    total_csv = pd.concat(total_csv_list, ignore_index=True)
    total_csv.to_csv('./FINAL_total_csv.csv', index=False)
