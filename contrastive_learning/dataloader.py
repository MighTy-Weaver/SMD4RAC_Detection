import pandas as pd
from torch.utils.data import Dataset


class AC_Triplet_Dataset(Dataset):
    def __init__(self, sample_frac=0.5, csv_path='./triplet.csv'):
        self.triplet_csv = pd.read_csv(csv_path, index_col=None).sample(frac=sample_frac).reset_index(drop=True)
        self.data = pd.read_csv('../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)

        self.X = self.data.drop(['Weekday', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time'], axis=1)
        self.Y = self.data['AC']
        self.X['Date'] = self.data['Time'].apply(lambda x: x.split(' ')[0])

    def __getitem__(self, index):
        if isinstance(index, int):
            anchor_room, anchor_date = self.triplet_csv.loc[index, 'anchor']
            pos_room, pos_date = self.triplet_csv.loc[index, 'pos']
            neg_room, neg_date = self.triplet_csv.loc[index, 'neg']
            return self.X[(self.X.Date == anchor_date) & (self.X.Location == anchor_room)] \
                       .drop(['Location', 'Date'], axis=1).to_numpy(dtype=float), \
                   self.X[(self.X.Date == pos_date) & (self.X.Location == pos_room)] \
                       .drop(['Location', 'Date'], axis=1).to_numpy(dtype=float), \
                   self.X[(self.X.Date == neg_date) & (self.X.Location == neg_room)] \
                       .drop(['Location', 'Date'], axis=1).to_numpy(dtype=float)
        elif isinstance(index, slice):
            raise NotImplementedError('Please implement the slice section')
        else:
            raise TypeError('Should call dataloader obj with int or slice')

    def __len__(self):
        return len(self.triplet_csv)


test = AC_Triplet_Dataset()
