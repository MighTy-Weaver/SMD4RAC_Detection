import pandas as pd
from torch.utils.data import Dataset


class AC_Dataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)
        print(list(self.data))
        self.X = self.data.drop(['Weekday', 'Location', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time'], axis=1)
        self.Y = self.data['AC']
        self.X['Date'] = self.data['Time'].apply(lambda x: x.split(' ')[0])
        dates = self.X['Date'].unique()
        print(self.X)
        print(dates)

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.X)


test = AC_Dataset()
