import sys
import warnings

import pandas as pd
from torch.utils.data import Dataset

sys.path.append('../')
from sparse_classification.utils import efficiency_dict
from sparse_classification.utils import normal_room_list

warnings.filterwarnings("ignore")


class AC_Normal_Dataset(Dataset):
    def __init__(self, mode='trn', test=False, cla=True):
        if mode not in ['trn', 'val']:
            raise NotImplementedError("mode must be either 'trn' or 'val'")

        self.test = test
        self.cla = cla

        self.data = pd.read_csv('../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)
        self.data = self.data[(self.data.AC > 0) & (self.data.Location.isin(efficiency_dict.keys()))].reset_index(
            drop=True).drop(['Weekday', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time'], axis=1)
        print(f"Normal DATALOADER: finished loading original data, total {len(self.data)}")

    def __getitem__(self, index):
        if self.cla:
            return self.data.drop(['Location'], axis=1).loc[index].to_numpy(), int(self.data.loc[
                                                                                       index, 'Location'] in normal_room_list)
        else:
            return self.data.loc[index].drop(['Location'], axis=1).to_numpy(), efficiency_dict[
                self.data.loc[index, 'Location']]

    def __len__(self):
        return len(self.data)
