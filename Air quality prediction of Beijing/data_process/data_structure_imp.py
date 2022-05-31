# @Time    : 2021/12/20 04:34
# @Author  : SY.M
# @FileName: data_structure_imp.py

import torch
import numpy as np
import pandas as pd
import math
import os
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, KFold

pd.set_option('display.max_columns', 100)  # a就是你要设置显示的最大列数参数
pd.set_option('display.max_rows', 20)  # b就是你要设置显示的最大的行数参数
pd.set_option('display.width', 10000)  # x就是你要设置的显示的宽度，防止轻易换行


class StationID_AQdata_Dict():
    """
    每个监测站ID与其空气质量数据组成的字典
    """

    def __init__(self, file_path):
        # 空气质量监测站id 其中1022监测站数据文件5种属性全部为None，因此选择丢弃
        self.station_id_list = [i for i in range(1001, 1022)] + [i for i in range(1023, 1037)]
        self.all_station_data = pd.read_csv(file_path)
        self.station_data_dict = self.build_dict()

    def build_dict(self):
        '''
        建立字典  key: 监测站id   value: 监测站空气质量数据
        :return:
        '''
        station_data_dict = dict()
        for id in self.station_id_list:
            station_data_dict[id] = self.all_station_data[self.all_station_data.station_id == id]
        return station_data_dict


class DistrictID_MLdata_Dict():
    """
    区域ID和其气象数据组成的字典
    """

    def __init__(self, file_path):
        self.district_id_list = [i for i in range(101, 117)]
        self.all_district_data = pd.read_csv(file_path)
        # print(self.all_district_data.weather.value_counts())
        self.district_data_dict = self.build_dict()
        # print(self.all_district_data.dtypes)

    def build_dict(self):
        """
        建立字典  key: 地区id  value:地区气象数据
        :return:
        """
        district_data_dict = dict()
        for id in self.district_id_list:
            district_data_dict[id] = self.all_district_data[self.all_district_data.id == id]

        return district_data_dict


class Complete_Broken_Stations_Dict():
    """
    将监测站空气质量数据与其所属的区域气象数据合并(外连接)  构成监测站ID与其空气质量和气象数据组成的字典
    """

    def __init__(self, SAQ_Dict, DML_Dict, index_file_path, pick_num):

        # pick_num = 3  # 选取几个最近站点

        district_belong = pd.read_csv(index_file_path, index_col='station_id').district_id[:36].to_dict()
        self.x_y = pd.read_csv(index_file_path)[0:37][['latitude', 'longitude']].to_numpy()
        # print(district_belong)
        # print(self.x_y)

        self.complete_broken_data = dict()
        self.closest_station_dict = self.get_closest(pick_num)

        for station_id in SAQ_Dict.keys():
            SAQ_Data = SAQ_Dict[station_id]
            DML_Data = DML_Dict[district_belong[station_id]]
            station_data = pd.merge(SAQ_Data, DML_Data, on='time', how='outer')

            # print(station_data.shape, end=' ')
            # print(station_data)

            self.complete_broken_data[station_id] = station_data

        # print()

        self.make_sequential()  # 使时间连续
        # self.find_max_len()  # 创建连续时间块

    def get_closest(self, pick_num):
        """
        为每个监测站选取最近的k个监测站并构建字典   key: 监测站id   value: 一个字典{key: 最近监测站id列表 value: 最近监测距离列表}
        :param pick_num: 选取最近pick_num个监测站
        :return:
        """
        closest_station_dict = dict()
        INF = 1000
        for i in range(36):
            distances = []
            for j in range(36):
                if i == j:
                    distances.append(INF)
                    continue
                dis = (self.x_y[i][0] - self.x_y[j][0]) ** 2 + (self.x_y[i][1] - self.x_y[j][1]) ** 2
                distances.append(dis)
            # print(distances)
            distances = torch.Tensor(distances)
            value, index = torch.topk(distances, k=pick_num, largest=False)
            # print(index, value)
            closest_station_dict[1001 + i] = {'station_id': index + 1001, 'distance': value}

        # print(closest_station_dict)

        return closest_station_dict

    def make_sequential(self):
        """
        建立连续时间数据（因为数据中时间可能不是连续的），并加入列表，列表中每个时间块数据都是时间连续的，
        用于之后在每个时间块上取样本
        :return:
        """
        for station_id, data in self.complete_broken_data.items():
            time_point = 0  # 原始数据第一条数据的hour
            start = 0
            end = 0
            sequential_list = []
            for hour in data.hour_x:
                # print(hour, time_point)
                if hour != time_point:
                    sequential_list.append(data[start:end])
                    start = end
                    time_point = hour
                end += 1
                time_point = (time_point + 1) % 24

            self.complete_broken_data[station_id] = sequential_list

    def find_max_len(self):
        length_list = []
        test_slice = []
        for station_id, data_list in self.complete_broken_data.items():
            after_slice_list = []
            for seq_index, sequentail_slice in enumerate(data_list):
                length = 0
                start = 0
                flag = True
                for i in range(0, len(sequentail_slice)):
                    if not flag:
                        break
                    # print(sequentail_slice.iloc[i])
                    data = sequentail_slice.iloc[i]
                    # print(type(data))
                    if not np.any(data.isnull()):
                        length += 1
                    else:
                        length_list.append(length)
                        if length >= 70:
                            # print('start:', start)
                            slice_prior = sequentail_slice[start - 24:start]
                            # print(slice_prior)
                            for m in range(slice_prior.shape[0]):
                                for n in range(3, slice_prior.shape[1]):
                                    # print(type(slice_prior.iloc[m, n]))
                                    # print(slice_prior.iloc[m, n], type(slice_prior.iloc[m, n]))
                                    if np.isnan(slice_prior.iloc[m, n]):
                                        back = start - 25 + m
                                        while np.isnan(sequentail_slice.iloc[back, n]):
                                            back -= 1
                                        slice_prior.iloc[m, n] = sequentail_slice.iloc[back, n]
                            slice = sequentail_slice[start:start + length]
                            slice = pd.concat([slice_prior, slice], axis=0)
                            slice = slice[:][['PM25_Concentration', 'PM10_Concentration',
                                              'NO2_Concentration', 'CO_Concentration',
                                              'O3_Concentration', 'SO2_Concentration',
                                              'weather', 'temperature', 'pressure', 'humidity',
                                              'wind_speed', 'wind_direction']]

                            before_slice = sequentail_slice[:start]
                            after_slice = sequentail_slice[start + length:]
                            print(station_id, seq_index, len(sequentail_slice), len(slice), len(before_slice),
                                  len(after_slice))
                            data_list[seq_index] = before_slice
                            # print(type(data_list[seq_index]))
                            after_slice_list.append(after_slice)
                            test_slice.append(np.array(slice))
                            flag = False
                        start = i + 1
                        length = 0
            if len(after_slice_list) != 0:
                self.complete_broken_data[station_id].extend(after_slice_list)
        print(pd.DataFrame({'length': length_list}).length.value_counts().sort_index(ascending=False))
        print(test_slice)
        np.save('./IMP sequential min_len=70 head AQ_ML.npy', test_slice)


def bulid_dataset(broken_data_dict, window_size, sliding_step, predict_length, train_pro=0.8, shuffle=True):

    global train_Y, test_Y
    station_dataset_dict = dict()

    data_dict = broken_data_dict.complete_broken_data

    sample_num = 0
    X = []  # 存储所有样本 不区分监测站
    Y = []  # 存储所有样本标签 不区分监测站
    for station_id, station_data_list in data_dict.items():
        X_list = []  # 临时存储一个监测站的所有样本
        Y_list = []  # 临时存储一个监测站的所有样本标签
        for sequential_slice in station_data_list:
            start = 0
            # data_len = sequential_slice.station_id.count()
            data_len = sequential_slice.time.count()
            # print(data_len)
            if data_len < window_size + predict_length:
                continue
            while (start + window_size + predict_length) <= data_len:
                # print((start + window_size + predict_length), data_len)
                sample_x = sequential_slice[start:start + window_size][['PM25_Concentration', 'PM10_Concentration',
                                                                        'NO2_Concentration', 'CO_Concentration',
                                                                        'O3_Concentration', 'SO2_Concentration',
                                                                        'weather', 'temperature', 'pressure',
                                                                        'humidity',
                                                                        'wind_speed', 'wind_direction']]
                sample_y = sequential_slice[start + window_size:start + window_size + predict_length][
                    ['PM25_Concentration', 'PM10_Concentration',
                     'NO2_Concentration', 'CO_Concentration',
                     'O3_Concentration', 'SO2_Concentration',
                     'weather', 'temperature', 'pressure', 'humidity',
                     'wind_speed', 'wind_direction']]
                start += sliding_step
                if np.any(sample_x.isnull()) or np.any(sample_y.isnull()):
                    # 存在NULL 删除样本
                    continue
                X_list.append(sample_x.to_numpy())
                Y_list.append(sample_y.to_numpy())

        print(f'stationID:{station_id}\t sample_num:{len(X_list)}')
        sample_num += len(X_list)
        if len(X_list) > 0:
            X.append(np.asarray(X_list))
            Y.append(np.asarray(Y_list))
        # X.append(torch.Tensor(X_list))
        # Y.append(torch.Tensor(Y_list))
        station_dataset_dict[station_id] = {'X': X_list, 'Y': Y_list}

    # build tensor dataset
    # X = torch.cat(X, dim=0).transpose(-1, -2)
    # Y = torch.cat(Y, dim=0)
    X = np.concatenate(X, axis=0).swapaxes(1, 2)
    Y = np.concatenate(Y, axis=0).reshape(-1, Y[0].shape[-1])
    if shuffle:
        shuffle_index = np.random.permutation(Y.shape[0])
        X = X[shuffle_index]
        Y = Y[shuffle_index]

    # 按照weather属性分层抽样，划分训练集测试集合
    Y_PD = pd.DataFrame(Y[:, 6])
    min_count = Y_PD.value_counts().min()
    sfk_indexset = StratifiedKFold(n_splits=7, shuffle=shuffle).split(X, Y[:, 6])
    for train_idx, test_idx in sfk_indexset:
        train_X, train_Y = X[train_idx], Y[train_idx]
        test_X, test_Y = X[test_idx], Y[test_idx]
        break

    # 按照比例随机划分训练集测试集合
    # train_end = math.ceil(sample_num * train_pro)
    # train_X = X[:train_end]
    # train_Y = Y[:train_end]
    # test_X = X[train_end:]
    # test_Y = Y[train_end:]

    print(f'All sample size:{X.shape}')
    print(f'trian dataset size:{train_X.shape}')
    print(f'test dataset size:{test_X.shape}')

    return train_X, train_Y, test_X, test_Y


def save_file(train_X, train_Y, test_X, test_Y, file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    else:
        print('覆盖原文件')
    np.save(file_path + 'train_X.npy', train_X)
    np.save(file_path + 'train_Y.npy', train_Y)
    np.save(file_path + 'test_X.npy', test_X)
    np.save(file_path + 'test_Y.npy', test_Y)
    # train_X = np.load(file_path)


def load_data(file_path):
    train_X = np.load(file_path + 'train_X.npy')
    train_Y = np.load(file_path + 'train_Y.npy')
    test_X = np.load(file_path + 'test_X.npy')
    test_Y = np.load(file_path + 'test_Y.npy')
    return train_X, train_Y, test_X, test_Y


class MyDataset(Dataset):
    def __init__(self, X, Y):
        super(MyDataset, self).__init__()
        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)
        self.sample_num = X.shape[0]
        self.feature_num = X.shape[1]
        self.time_len = X.shape[2]

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return self.sample_num


if __name__ == '__main__':
    file_path_1 = 'HERE IS THE ROOT OF RAW DATA/airquality.csv'
    file_path_2 = 'HERE IS THE ROOT OF RAW DATA/meteorology.csv'
    file_path_3 = 'HERE IS THE ROOT OF RAW DATA/station.csv'
    a = StationID_AQdata_Dict(file_path=file_path_1)
    b = DistrictID_MLdata_Dict(file_path=file_path_2)
    c = Complete_Broken_Stations_Dict(a.station_data_dict, b.district_data_dict, file_path_3, pick_num=3)

    window_size = 24
    sliding_step = 1
    predict_length = 1
    train_X, train_Y, test_X, test_Y = bulid_dataset(c, window_size=window_size, sliding_step=sliding_step,
                                                     predict_length=predict_length)
    save_file(train_X, train_Y, test_X, test_Y, file_path=f'./IMP ws={window_size} ss={sliding_step} pl={predict_length} 0.8 Tianjin AQ_ML/')
