import pandas as pd

data = pd.read_csv('./prediction.csv')

replaced_room_dict = {
    324: '2022/4/28',
    619: '2022/4/28',
    916: '2022/4/28',
    911: '2022/6/15',
    632: '2022/6/24',
    1004: '2022/6/24',
    301: '2022/7/14',
    1014: '2022/7/14',
    304: '2022/8/23',
    633: '2022/8/23',
    805: '2022/8/23',
    909: '2022/8/23',
    1006: '2022/8/23',
    1010: '2022/8/23',
    306: '2022/9/14',
    635: '2022/9/23'
}

data_not_replaced = data[~data.room.isin(replaced_room_dict.keys())]
# number of data_not_replaced whose prediction is higher than 0.5
print(len(data_not_replaced[data_not_replaced.pred > 0.5]))
print(len(data_not_replaced))