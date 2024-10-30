import warnings
import numpy as np
import pandas as pd
from mindspore.dataset import Dataset, GeneratorDataset
from mindspore.dataset.transforms import Compose, Map
from mindspore.dataset.vision import ToTensor
from mindspore import Tensor
from sklearn.preprocessing import StandardScaler

from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


class CIDatasetBenchmark:
    def __init__(self, root_path='dataset', flag='train', input_len=None, pred_len=None,
                 data_type='custom', scale=True, timeenc=1, freq='h', stride=1, subset_rand_ratio=1.0):
        self.subset_rand_ratio = subset_rand_ratio
        self.input_len = input_len
        self.pred_len = pred_len
        self.seq_len = input_len + pred_len
        self.timeenc = timeenc
        self.scale = scale
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_type = data_type

        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1

        self.root_path = root_path
        self.dataset_name = self.root_path.split('/')[-1].split('.')[0]

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = self.root_path

        # Load data
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)

        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        # Data splitting logic
        data_len = len(df_raw)
        if self.data_type == 'custom':
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]
        else:
            # Add handling for other dataset types if needed
            raise ValueError('Unsupported data type: {}'.format(self.data_type))

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data = df_raw.values

        # Scale data
        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        # Prepare timestamps
        if self.timeenc == 0:
            df_stamp = df_raw[['date']]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_raw.date.values), freq='h')
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError('Unknown timeenc: {}'.format(self.timeenc))

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]

        self.n_var = self.data_x.shape[-1]
        self.n_timepoint = len(self.data_x) - self.input_len - self.pred_len + 1

    def get_dataset(self):
        data_generator = self.__data_generator__()
        return GeneratorDataset(data_generator, ["seq_x", "seq_y", "seq_x_mark", "seq_y_mark"],
                                shuffle=True)

    def __data_generator__(self):
        for index in range(self.__len__()):
            yield self.__getitem__(index)

    def __getitem__(self, index):
        if self.set_type == 0:
            index = index * self.internal
        c_begin = index // self.n_timepoint  # select variable
        s_begin = index % self.n_timepoint   # select start time
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end, c_begin:c_begin + 1]
        seq_y = self.data_y[r_begin:r_end, c_begin:c_begin + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return Tensor(seq_x, dtype=mindspore.float32), Tensor(seq_y, dtype=mindspore.float32), \
               Tensor(seq_x_mark, dtype=mindspore.float32), Tensor(seq_y_mark, dtype=mindspore.float32)

    def __len__(self):
        if self.data_type == 'wind':
            return len(self.data_x) - self.input_len - self.pred_len + 1
        if self.set_type == 0:
            return max(int(self.n_var * self.n_timepoint * self.subset_rand_ratio), 1)
        else:
            return int(self.n_var * self.n_timepoint)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class CIAutoRegressionDatasetBenchmark(CIDatasetBenchmark):
    def __init__(self, root_path='dataset', flag='train', input_len=None, label_len=None, pred_len=None,
                 data_type='custom', scale=True, timeenc=1, freq='h', stride=1, subset_rand_ratio=1.0):
        self.label_len = label_len
        super().__init__(root_path=root_path, flag=flag, input_len=input_len, pred_len=pred_len,
                         data_type=data_type, scale=scale, timeenc=timeenc, freq=freq, stride=stride,
                         subset_rand_ratio=subset_rand_ratio)

    def __getitem__(self, index):
        if self.set_type == 0:
            index = index * self.internal
        c_begin = index // self.n_timepoint  # select variable
        s_begin = index % self.n_timepoint   # select start time
        s_end = s_begin + self.input_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, c_begin:c_begin + 1]
        seq_y = self.data_y[r_begin:r_end, c_begin:c_begin + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return Tensor(seq_x, dtype=mindspore.float32), Tensor(seq_y, dtype=mindspore.float32), \
               Tensor(seq_x_mark, dtype=mindspore.float32), Tensor(seq_y_mark, dtype=mindspore.float32)
