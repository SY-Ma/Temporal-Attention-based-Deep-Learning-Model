# @Time    : 2022/02/14 25:03
# @Author  : SY.M
# @FileName: early_stop.py


import copy
import os
import torch

class Eearly_stop():
    def __init__(self, patience=5, saved_path=None, save_model_flag=True):
        self.saved_path = saved_path
        self.origin_patience = patience
        self.patience = patience
        self.max_acc = -10000000000
        self.min_e = 10000000000
        self.net = None
        self.score = None
        self.save_model_flag = save_model_flag

    def judge_acc(self, acc, net, score=None):
        if acc > self.max_acc:
            self.max_acc = acc
            self.patience = self.origin_patience
            self.net = copy.deepcopy(net)
            # self.score = copy.deepcopy(score.detach())
        else:
            self.patience -= 1
            if self.patience <= 0:
                print('\033[0;34mpatience == 0, stop training!\033[0m\t\t')
                return True  # 停止训练
        print('patience = ', self.patience)

        return False  # 继续训练

    def judge_e(self, e, net, score):
        if e < self.min_e:
            self.min_e = e
            self.patience = self.origin_patience
            self.net = copy.deepcopy(net)
            # self.score = copy.deepcopy(score.detach())
        else:
            self.patience -= 1
            if self.patience <= 0:
                print('\033[0;34mpatience == 0, stop training!\033[0m\t\t')
                return True  # 停止训练
        print('patience = ', self.patience)

    def __call__(self, acc, net, type, score=None):
        if type == 'acc':
            return self.judge_acc(acc, net, score)
        elif type == 'e':
            return self.judge_e(acc, net, score)

    def save_model(self, acc, model_name):
        if self.save_model_flag:
            if not os.path.exists(self.saved_path['root'] + '/' + model_name):
                os.makedirs(self.saved_path['root'] + '/' + model_name)

            torch.save(self.net, self.saved_path['root'] + '/' + model_name + '/random_seed=' + str(self.saved_path['random_seed']) + ' ' +
                                                 self.saved_path['imp_feature'] + ' ' + str(round(acc, 4)) + ' .pkl')

            print('保存成功！')