# Temporal Attention-based Deep Learning Model

***An Attentnion-based Deep Learning Model for filling missing values in Multivariate Air Quality Data***

## Abstract
>空气污染严重影响着人类的身体健康与社会的可持续发展，但传感器获取的多元变量空气质量数据往往存在缺失值，这为数据的分析与处理带来了困扰。目前许多对某一种空气成分变化的分析方法，只依赖于此属性的时间数据与空间数据，忽略了在相同时间区间内其他若干空气成分对此属性变化趋势的影响，且在离散型缺失数据的填充上难以达到理想的效果。本研究提出了一种时间注意力深度学习模型(TAM)，该模型使用注意力机制来关注不同时间戳之间的相关性与不同特征时间序列之间的相关性，并结合短期历史数据，来填充多元变量空气质量数据中的缺失读数。本文使用北京市的空气质量数据对所提出的方法进行评估，实验结果表明TAM相比较于其他十种基线模型具有优势。\
>**关键词**: 空气质量、缺失值填充、Attention机制、深度学习

>Air pollution has a serious impact on human health and the sustainable development of society, but the multivariate air quality data obtained by sensors generally have missing values which causes problems in data analysis and processing. Many existing methods in dealing with a certain air component only depend on its temporal and spatial information which ignores the interaction between different attributes, moreover they are difficult to achieve ideal results in the prediction of discrete missing data. This paper proposes a temporal attention-based deep learning model (TAM), which uses attention mechanism to focus on the correlation between different time stamps and the correlation between different feature time series, it also combines short-term historical data to fill in missing readings in multivariate air quality data. This paper uses the real air quality data of Beijing to evaluate our method, and the experimental results show that TAM has advantages over the other ten baseline models. \
**Key words**: air quality; missing data imputation; attention mechanism; deep learning


Please cite as:  
`还没投:）`  

## Introduction
如今，人们越来越关注空气污染问题，因为它时刻威胁着人类的身体健康与社会的可持续发展，因此人们在城市中建立了越来越多的监测站，来不断获取空气质量数据与气象数据等，这为人们分析污染来源、探究污染主要成分、预测空气质量提供了数据基础。但是，由于监测设备的停机维护、损坏、通信错误、意外中断(如停电)等原因，导致传感器监测得到的数据含有缺失数值。缺失数据不仅会影响实时的污染物数值监测，还会为数据分析和污染物浓度预测带来干扰。  
空气质量数据的缺失模型如图:  
![Error](https://raw.githubusercontent.com/SY-Ma/Temporal-Attention-based-Deep-Learning-Model/main/images/%E7%BC%BA%E5%A4%B1%E7%B1%BB%E5%9E%8B%E5%B1%95%E7%A4%BA%E5%9B%BE%202.png)

## Framework
本文提出了一种时间注意力深度学习模型(a **T**emporal **A**ttention-based Deep Learning **M**odel **TAM**)来处理多元变量空气质量数据中的缺失读数，其使用Attention机制来关注不同时间戳之间的相关性与不同特征时间序列之间的相关性，并结合短期历史数据，得到最终的预测结果。模型结构如图：  
![Error](https://github.com/SY-Ma/Temporal-Attention-based-Deep-Learning-Model/blob/31ad7417d3c549ddc6f0530d037068989a76706d/images/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%E5%9B%BE%20%E5%BD%A9%E8%89%B2.png)

## Code
- [TAM for Discrete Data](https://gitee.com/SY-M/temporal-attention-based-deep-learning-model/blob/master/Air%20quality%20Missing%20Data%20Imputation/model/imp_module_cla_TAM.py)
- [TAM for Consecutive Data](https://gitee.com/SY-M/temporal-attention-based-deep-learning-model/blob/master/Air%20quality%20Missing%20Data%20Imputation/model/imp_module_pre_TAM.py)  

## Result
- [Sample Dataset Result](result/Block%20Dataset%20Result.xlsx)
- [Block Dataset Result](result/Block%20Dataset%20Result.xlsx)

## Demo
- 运行[run_early_stop.py](https://gitee.com/SY-M/temporal-attention-based-deep-learning-model/blob/master/Air%20quality%20Missing%20Data%20Imputation/run_early_stop.py)进行Sample Dataset上的模型训练与测试。
- 运行[run_block_imp.py](https://gitee.com/SY-M/temporal-attention-based-deep-learning-model/blob/master/Air%20quality%20Missing%20Data%20Imputation/run_block_imp.py)进行Block Dataset上的测试。

```
import torch
from torch.utils.data import DataLoader
from data_process import data_structure_imp as dsi
from model.imp_module_pre_TAM import Imp_Single as Pre_Model_TAM
from model.imp_module_cla_TAM import Imp_Single as Cla_Model_TAM
from model.loss_function import Classify_Loss, Predict_Loss
from utils.random_seed import setup_seed
from utils.early_stop import Eearly_stop

# 0. load data and create dataloader
file_path = 'data_process/' + 'Imp_dataset' + '/'
train_X, train_Y, test_X, test_Y = dsi.load_data(file_path=file_path)
val_X = train_X[4000:]  
val_Y = train_Y[4000:]
train_X = train_X[:4000]
train_Y = train_Y[:4000]
train_dataset = dsi.MyDataset(X=train_X, Y=train_Y)
val_dataset = dsi.MyDataset(X=val_X, Y=val_Y)
test_dataset = dsi.MyDataset(X=test_X, Y=test_Y)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 1. choose feature to train and test
feature_idx = {'PM25_Concentration': (0, 1140.0 - 1.0, None), 'PM10_Concentration': (1, 1472.0 - 0.1, None),
               'NO2_Concentration': (2, 499.7 - 0.0, None),
               'CO_Concentration': (3, 27.175 - 0.0, None), 'O3_Concentration': (4, 500.0 - 0.0, None),
               'SO2_Concentration': (5, 986.0 - 0.0, None),
               'weather': (6, None, 17), 'temperature': (7, 41.0 - (-27), None), 'pressure': (8, 1042.0 - 745.7, None),
               'humidity': (9, 100.0 - 0.0, None),
               'wind_speed': (10, 90.0 - 0.0, None), 'wind_direction': (11, None, 25)}
imp_feature = 'PM25_Concentration'
# imp_feature = 'PM10_Concentration'
# imp_feature = 'NO2_Concentration'
[...]

# 2. build model
Model = None
if imp_feature in ['weather', 'wind_direction']:
    Model = Cla_Model_TAM
    judge_flag = 'acc'
else:
    Model = Pre_Model_TAM
    judge_flag = 'e'

net = Model(num_feature=num_feature, time_len=time_len, hidden_len=hidden_len, q=q, k=k, v=v, h=h, N=N,
                 temp_len=temp_len,
                 mask=mask, is_stand=is_stand, num_classes=feature_idx[imp_feature][-1]).to(DEVICE)

# 3. build loss function and optimizer
if imp_feature in ['weather', 'wind_direction']:
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_function = Classify_Loss()
else:
    optimizer = torch.optim.Adam([{'params': net.input_layer_feature.parameters()},
                                  {'params': net.input_layer_time.parameters()},
                                  {'params': net.encoder_stack_feature.parameters()},
                                  {'params': net.encoder_stack_time.parameters()},
                                  {'params': net.output_linear.time_linear.parameters()},
                                  {'params': net.output_linear.feature_linear.parameters()},
                                  {'params': net.output_linear.weight_linear, 'lr': 1e-2}], lr=LR)
    loss_function = Predict_Loss()

# 4. train and test
[...]

```

## Reference
```bibtex
@inproceedings{2015Forecasting,
  title={Forecasting Fine-Grained Air Quality Based on Big Data},
  author={ Yu, Z.  and  Yi, X.  and  Ming, L.  and  Li, R.  and  Shan, Z. },
  booktitle={the 21th ACM SIGKDD International Conference},
  year={2015},
}
@article{2016ST,
  title={ST-MVL: filling missing values in geo-sensory time series data},
  author={ Yi, X.  and  Yu, Z.  and  Zhang, J.  and  Li, T. },
  booktitle = {Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence},
  year={2016},
}
@article{RN3,
   author = {Ngueilbaye, Alladoumbaye and Wang, Hongzhi and Mahamat, Daouda Ahmat and Junaidu, Sahalu B.},
   title = {Modulo 9 model-based learning for missing data imputation},
   journal = {Applied Soft Computing},
   volume = {103},
   ISSN = {15684946},
   DOI = {10.1016/j.asoc.2021.107167},
   year = {2021},
   type = {Journal Article}
}
```
