# @Time    : 2022/02/14 25:03
# @Author  : SY.M
# @FileName: run_early_stop.py


import torch
from torch.utils.data import DataLoader
from data_process import data_structure_imp as dsi
from model.imp_module_pre_TAM import Imp_Single as Pre_Model_TAM
from model.imp_module_cla_TAM import Imp_Single as Cla_Model_TAM
from model.loss_function import Classify_Loss, Predict_Loss
from utils.random_seed import setup_seed
from utils.early_stop import Eearly_stop


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
# imp_feature = 'CO_Concentration'
# imp_feature = 'O3_Concentration'
# imp_feature = 'SO2_Concentration'
# imp_feature = 'weather'
# imp_feature = 'temperature'
# imp_feature = 'pressure'
# imp_feature = 'humidity'
# imp_feature = 'wind_speed'
# imp_feature = 'wind_direction'

print(f'预测属性:{imp_feature}')
print('数据读取中...')
file_path = 'data_process/' + 'Imp_dataset' + '/'
train_X, train_Y, test_X, test_Y = dsi.load_data(file_path=file_path)
val_X = train_X[4000:]  # 验证集
val_Y = train_Y[4000:]
train_X = train_X[:4000]  # 训练集
train_Y = train_Y[:4000]
print(f'trian dataset size:{train_X.shape}')
print(f'val dataset size:{val_X.shape}')
print(f'test dataset size:{test_X.shape}')

train_dataset = dsi.MyDataset(X=train_X, Y=train_Y)
val_dataset = dsi.MyDataset(X=val_X, Y=val_Y)
test_dataset = dsi.MyDataset(X=test_X, Y=test_Y)

seed = 30
setup_seed(seed)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'use device:{DEVICE}\t\trandom_seed:{seed}')
save_model_flag = True  # 是否保存模型与模型参数到.pkl文件

EPOCH = 10000
BATCH_SIZE = 512
test_interval = 1

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 超参 以下为模型使用超参 各个属性使用相同的超参
num_feature = train_dataset.feature_num
time_len = train_dataset.time_len
hidden_len = 512  # d_model
q = 8
k = q
v = 8
h = 512
N = 1
temp_len = 256
mask = False
is_stand = True
LR = 1e-4
early_stop = Eearly_stop(patience=50, save_model_flag=save_model_flag,
                         saved_path={'root': 'saved_model', 'random_seed': seed, 'imp_feature': imp_feature})
print(f'超参: Batch={BATCH_SIZE} hidden_len={hidden_len} q=k=v={q} h={h} N={N}')
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

print('开始训练...')


def train():
    for epoch_idx in range(EPOCH):
        net.train()
        epoch_loss = 0
        for batch_idx, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            out, score_feature, score_time = net(x.to(DEVICE), 'train', feature_idx[imp_feature][0])
            loss = loss_function(out, y[:, feature_idx[imp_feature][0]].to(DEVICE))
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch:{epoch_idx}\t\tTrain_Loss:{epoch_loss / train_X.shape[0]}')

        if epoch_idx % test_interval == 0:
            if feature_idx[imp_feature][-1] is None:
                p, e = test_p(dataloader=val_dataloader, dataset_type='val', test_net=net)
                test_p(dataloader=train_dataloader, dataset_type='train', test_net=net)
                if early_stop(e, net, type=judge_flag):
                    p, e = test_p(dataloader=test_dataloader, dataset_type='test', test_net=early_stop.net)
                    test_p(dataloader=val_dataloader, dataset_type='val', test_net=early_stop.net)
                    test_p(dataloader=train_dataloader, dataset_type='train', test_net=early_stop.net)
                    if save_model_flag:
                        early_stop.save_model(acc=e, model_name='TAM')
                    return
            else:
                acc = test_acc(dataloader=val_dataloader, dataset_type='val', test_net=net)
                test_acc(dataloader=train_dataloader, dataset_type='train', test_net=net)
                if early_stop(acc, net, type=judge_flag):
                    acc = test_acc(dataloader=test_dataloader, dataset_type='test', test_net=early_stop.net)
                    test_acc(dataloader=val_dataloader, dataset_type='val', test_net=early_stop.net)
                    test_acc(dataloader=train_dataloader, dataset_type='train', test_net=early_stop.net)
                    if save_model_flag:
                        early_stop.save_model(acc=acc, model_name='TAM')
                    return


def test_acc(dataloader, dataset_type, test_net):
    with torch.no_grad():
        net.eval()
        correct = 0
        total = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            total += x.shape[0]
            out, score_feature, score_time = test_net(x.to(DEVICE), 'test', feature_idx[imp_feature][0])
            _, imp_index = torch.max(out, dim=-1)
            if imp_feature == 'wind_direction':
                y = y[:, feature_idx[imp_feature][0]]
                correct += torch.sum(torch.eq(imp_index, y.long().to(DEVICE))).item()
            else:
                correct += torch.sum(torch.eq(imp_index, y[:, feature_idx[imp_feature][0]].long().to(DEVICE))).item()
        accuracy_on_test = round(correct / total, 4) * 100
        if dataset_type == 'test' or dataset_type == 'val':
            print(f'accuracy on {dataset_type}:\033[0;34m{accuracy_on_test}%\033[0m', end='\t\t')
        else:
            print(f'accuracy on {dataset_type}:{accuracy_on_test}%', end='\t\t')

        return accuracy_on_test


def test_p(dataloader, dataset_type, test_net):
    with torch.no_grad():
        net.eval()
        # total = 0
        out_all = []
        y_all = []
        for batch_idx, (x, y) in enumerate(dataloader):
            # total += x.shape[0]
            out, score_feature, score_time = test_net(x.to(DEVICE), 'test', feature_idx[imp_feature][0])
            out_all.append(out)
            y_all.append(y[:, feature_idx[imp_feature][0]].float().to(DEVICE))
        out = torch.cat(out_all, dim=0).squeeze()
        y = torch.cat(y_all, dim=0)
        p = torch.mean(1 - torch.abs(out - y) / y).item()
        e = torch.mean(torch.abs(out - y)).item()
        gmad = torch.mean(torch.abs(out - y) / feature_idx[imp_feature][1]).item()
        mae = loss_function(out.unsqueeze(-1), y).item() / out.shape[0]
        if dataset_type == 'test' or dataset_type == 'val':
            print(
                f'accuracy on {dataset_type}:\033[0;34mp={round(p, 4)}\033[0m\te={round(e, 4)}\tg={round(gmad, 6)}\tmae={round(mae, 4)}',
                end='\t\t')
        else:
            print(
                f'accuracy on {dataset_type}:p={round(p, 4)}\te={round(e, 4)}\tg={round(gmad, 6)}\tmae={round(mae, 4)}',
                end='\t\t')

        return p, e


if __name__ == '__main__':
    train()
