# @Time    : 2022/03/17 22:06
# @Author  : SY.M
# @FileName: run_block_imp.py


import numpy as np
import pandas as pd
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from utils.random_seed import setup_seed
from data_process import data_structure_imp as dsi
from sklearn.naive_bayes import GaussianNB

feature_idx = {'PM25_Concentration': (0, 1140.0 - 1.0, None), 'PM10_Concentration': (1, 1472.0 - 0.1, None),
               'NO2_Concentration': (2, 499.7 - 0.0, None),
               'CO_Concentration': (3, 27.175 - 0.0, None), 'O3_Concentration': (4, 500.0 - 0.0, None),
               'SO2_Concentration': (5, 986.0 - 0.0, None),
               'weather': (6, None, 17), 'temperature': (7, 41.0 - (-27), None), 'pressure': (8, 1042.0 - 745.7, None),
               'humidity': (9, 100.0 - 0.0, None),
               'wind_speed': (10, 90.0 - 0.0, None), 'wind_direction': (11, None, 25)}


class Data_Centre():
    def __init__(self, path: str, missing_rate: float, used_model: str, random_seed: int):
        '''
        构造函数
        :param path: 块缺失数据文件路径
        :param missing_rate: 块缺失比率
        :param used_model: 使用模型名称
        '''
        self.random_seed = random_seed  # 用于选择模型的随机种子
        self.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.missing_rate = missing_rate  # 块缺失率
        self.used_model = used_model  # 使用的模型名称，用于寻找对应模型的.pkl文件
        self.unbroken_list = np.load(path, allow_pickle=True)  # 未缺失数据，即True Label，用于计算损失
        self.dynamic_data = np.load(path, allow_pickle=True)  # 待填充数据块
        self.mask_list = self.build_mask_map()  # 缺失位置表示矩阵
        self.feature_model_dict = self.get_model_dict()  # 每种属性填充模型的.pkl文件组成的字典

    def build_mask_map(self):
        '''
        创建缺失矩阵，反应数据矩阵是否缺失数值，是哪一类缺失
        0：Observed Value   1：General Missing    2：Temporal Block Missing
        :return: mask_list
        '''
        mask_list = []
        for dataframe in self.unbroken_list:
            dataframe = torch.Tensor(np.array(dataframe.astype(np.float32)))
            start_area = torch.zeros(24, 12)
            missing_area = torch.zeros_like(dataframe[24:])

            mask = torch.rand_like(missing_area)
            mask = torch.where(mask <= self.missing_rate, torch.ones(1), torch.zeros(1))  # 不缺失为0

            for col in range(mask.shape[1]):
                for line in range(mask.shape[0]-1):
                    if mask[line, col] == 1:
                        # 该位置为缺失值
                        if mask[line+1, col] > 0:
                            mask[line, col] = 2
                            mask[line+1, col] = 2
            mask = torch.cat([start_area, mask], dim=0)
            mask_list.append(mask)
        return mask_list

    def missing_describe(self):
        '''
        各种缺失模式的缺失率
        :return:
        '''
        general_missing_num = np.zeros(12)
        all_missing_num = np.zeros(12)
        block_missing_num = np.zeros(12)
        all_data_sum = np.zeros(12)
        for slice_idx, mask in enumerate(self.mask_list):
            mask = mask.numpy()
            all_data_sum += mask.shape[0] - 24
            all_missing_num += np.sum(mask > 0, axis=0)
            general_missing_num += np.sum(mask == 1, axis=0)
            block_missing_num += np.sum(mask == 2, axis=0)
        print(f'all missing rate:{all_missing_num / all_data_sum}\r\n'
              f'general missing rate:{general_missing_num / all_data_sum}\r\n'
              f'block missing rate:{block_missing_num / all_data_sum}\r\n')

    def get_model_dict(self):
        '''
        获取属性标号何其对应的pkl模型字典
        :return:
        '''
        path_dict = {}

        for index, feature in enumerate(feature_idx.keys()):
            path = f'saved_model/{self.used_model}/{self.get_model_file_name(self.random_seed, feature)}'
            print(path)
            if os.path.exists(path):
                path_dict[index] = torch.load(path, map_location=torch.device('cuda:0'))
            # path_dict[index] = path
            else:
                path_dict[index] = None
        return path_dict

    def get_model_file_name(self, random_seed, imp_feature):
        '''
        获得所用的模型对应随机种子下的所有属性预测模型.pkl文件名
        :param used_model: 使用的模型
        :param random_seed: 随机种子
        :param imp_feature: 预测的特征
        :return: f 文件名
        '''
        path = f'saved_model/{self.used_model}'
        if not os.path.exists(path):
            return
        for f in os.listdir(path):
            if f'random_seed={self.random_seed}' in f and imp_feature in f:
                return f

    def imputation(self):
        '''
        按行查找，若该行存在缺失值，则遍历该行，查找缺失值所在属性，并调用对应的模型进行预测，并将预测值填入原矩阵
        :return:
        '''
        for slice_idx, mask in enumerate(self.mask_list):
            for line in range(24, mask.shape[0]):
                if torch.sum(mask[line, :]) > 0:
                    # 说明该行存在缺失值
                    for col, value in enumerate(mask[line, :]):
                        if value > 0:
                            # 说明该属性为缺失值
                            self.update_dynamic_data(slice_idx=slice_idx, line=line, col=col)

    def update_dynamic_data(self, slice_idx, line, col):
        '''
        划分输入数据，根据特征选择模型，进行预测，将预测的数值插入到动态数据中
        :param slice_idx: 块编号
        :param line: 某一块中的数据所在行数
        :param col: 某一块中的数据所在列数(特征标号)
        :return:
        '''
        model = self.feature_model_dict[col]
        block_data = self.dynamic_data[slice_idx]
        input_data = block_data[line-24:line, :]
        input_data = torch.from_numpy(input_data).float().unsqueeze(0).transpose(-1, -2)
        out, _, _ = model(input_data.to(self.DEVICE), stage='test', feature_idx=col)
        if col not in [6, 11]:
            # 6和11是分类 输出数据不是预测值
            pre = out.item()
        else:
            _, pre = torch.max(out, dim=-1)
            pre = pre.item()
        self.dynamic_data[slice_idx][line, col] = pre  # 将预测值填入


if __name__ == '__main__':
    path = 'data_process/IMP sequential min_len=70 AQ_ML.npy'

    start_seed, end_seed = 30, 31
    seed_num = end_seed - start_seed
    for seed in range(start_seed, end_seed):
        setup_seed(seed)
        DC = Data_Centre(path=path, missing_rate=0.1, used_model='TAM', random_seed=30)
        DC.missing_describe()
        DC.imputation()

        # 在某一随机种子下填充完成，可进行指标计算与可视化等操作....
