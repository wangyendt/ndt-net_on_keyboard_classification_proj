#!/usr/bin/python
# coding: utf-8
# @Time    : 2018/8/17 12:48
# @Author  : Ye Wang (Wane)
# @Email   : y.wang@newdegreetech.com
# @File    : ndt_net_torch.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


class NDT_Net(nn.Module):
    def __init__(self):
        super(NDT_Net, self).__init__()
        self.l1 = nn.Linear(6, 20)
        self.bn1 = nn.BatchNorm1d(20)
        self.l2 = nn.Linear(20, 20)
        self.bn2 = nn.BatchNorm1d(20)
        self.l3 = nn.Linear(20, 6)
        init.normal_(self.l1.weight, mean=0, std=0.01)
        init.normal_(self.l2.weight, mean=0, std=0.01)
        init.normal_(self.l3.weight, mean=0, std=0.01)

    def forward(self, x):
        # print('input', x.cpu().numpy()[0, :])
        x = F.relu(self.bn1(self.l1(x)))
        # print('l1', x.detach().cpu().numpy()[0, :])
        x = F.relu(self.bn2(self.l2(x)))
        # print('l2', x.detach().cpu().numpy()[0, :])
        # x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        out = F.softmax(self.l3(x))
        return out


def train(loader, epochs, learning_rate):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                 learning_rate,
                                 weight_decay=0.001
                                 )
    criterion = nn.BCELoss()

    cost = []
    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(loader):
            bx, by = Variable(batch_x).cuda(), Variable(batch_y).cuda()
            optimizer.zero_grad()
            y_pred = model(bx)
            loss = criterion(y_pred, by)
            cost.append(loss)
            loss.backward()
            optimizer.step()
        print('Epoch: ', epoch, ' | loss: ', loss.item())
    plt.plot(cost)
    plt.show()


def my_validate(x_test, y_test, x_real):
    mdl_dict = model.state_dict()
    for var in mdl_dict:
        np_var = mdl_dict[var].cpu().numpy()
        np.savetxt(var, np_var, '%f')
    exit()
    l1_w = mdl_dict['l1.weight'].cpu().numpy()
    l1_b = mdl_dict['l1.bias'].cpu().numpy()
    bn1_w = mdl_dict['bn1.weight'].cpu().numpy()
    bn1_b = mdl_dict['bn1.bias'].cpu().numpy()
    bn1_running_mean = mdl_dict['bn1.running_mean'].cpu().numpy()
    bn1_running_var = mdl_dict['bn1.running_var'].cpu().numpy()
    l2_w = mdl_dict['l2.weight'].cpu().numpy()
    l2_b = mdl_dict['l2.bias'].cpu().numpy()
    bn2_w = mdl_dict['bn2.weight'].cpu().numpy()
    bn2_b = mdl_dict['bn2.bias'].cpu().numpy()
    bn2_running_mean = mdl_dict['bn2.running_mean'].cpu().numpy()
    bn2_running_var = mdl_dict['bn2.running_var'].cpu().numpy()
    l3_w = mdl_dict['l3.weight'].cpu().numpy()
    l3_b = mdl_dict['l3.bias'].cpu().numpy()
    # print('input', x_test[0, :])
    y_pred = np.dot(x_test, l1_w.T) + l1_b.T
    y_pred = (y_pred - bn1_running_mean) / np.sqrt(bn1_running_var + 1e-5)
    y_pred = bn1_w * y_pred + bn1_b
    y_pred[y_pred < 0] = 0
    # print('l1', y_pred[0, :])
    y_pred = np.dot(y_pred, l2_w.T) + l2_b.T
    y_pred = (y_pred - bn2_running_mean) / np.sqrt(bn2_running_var + 1e-5)
    y_pred = bn2_w * y_pred + bn2_b
    y_pred[y_pred < 0] = 0
    # print('l2', y_pred[0, :])
    y_pred = np.dot(y_pred, l3_w.T) + l3_b.T
    y_pred = np.apply_along_axis(
        lambda x: np.exp(x) / np.sum(np.exp(x)),
        0, y_pred
    )
    y_pred = y_pred.argmax(axis=1)
    y_test = y_test.argmax(axis=1)
    # print(y_pred)
    # print(y_test)
    print('Accuracy: ', sum(y_pred == y_test) / len(y_test))


def validate(x_test, y_test, x_real):
    model.eval()
    y_pred = model(Variable(torch.Tensor(x_test)).cuda()).cpu()

    y_pred = y_pred.detach().numpy()
    y_pred = y_pred.argmax(axis=1)
    y_test = y_test.argmax(axis=1)
    print(y_pred)
    print(y_test)
    print('Accuracy: ', sum(y_pred == y_test) / len(y_test))
    # mistake_indices = np.where(y_pred != y_test)[0]
    # # mistake_indices = range(100, 115)
    # num2key = {
    #     0: 'key1',
    #     1: 'key2',
    #     2: 'key3',
    #     3: 'key4',
    #     4: 'key5',
    #     5: 'No key'
    # }
    # for mi in mistake_indices:
    #     fig = plt.figure()
    #     fig.set_size_inches(60, 10)
    #     plt.plot(x_real[mi])
    #     # print(np.shape(x_test))
    #     # print(np.shape(x_test[mi][np.newaxis, :]))
    #     plt.text(2, 3, str(model(Variable(torch.Tensor(x_test[mi][np.newaxis, :])).cuda()).cpu()))
    #     plt.title('y_pred: {}, y_test: {}'.format(num2key[y_pred[mi]], num2key[y_test[mi]]))
    #     plt.show()


if __name__ == '__main__':
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = 'Consolas'
    torch.manual_seed(1)
    import DataLoader as dl

    X_train, y_train, X_test, y_test, X_real_train, X_real_test = dl.load_data('data', 30, 0.7)
    data_set = Data.TensorDataset(torch.Tensor(X_train),
                                  torch.Tensor(y_train))
    loader = Data.DataLoader(
        dataset=data_set,
        # batch_size=X_train.shape[0],
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    model = NDT_Net().cuda()
    # for k in model.state_dict():
    #     print(k)
    # print(model.state_dict()['l1.weight'])
    # exit()
    # train(loader, 20, 0.01)
    # torch.save(model.state_dict(), 'model.pkl')
    # torch.save(model.state_dict(), 'model_without_affine.pkl')
    # np.savetxt('w1',model.state_dict()['w1'])
    model.load_state_dict(torch.load('model.pkl'))
    # model.load_state_dict(torch.load('model_without_affine.pkl'))
    validate(X_train, y_train, X_real_train)
    validate(X_test, y_test, X_real_test)
    my_validate(X_train, y_train, X_real_train)
    my_validate(X_test, y_test, X_real_test)
