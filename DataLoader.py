#!/usr/bin/python
# coding: utf-8
# @Time    : 2018/8/17 12:39
# @Author  : Ye Wang (Wane)
# @Email   : y.wang@newdegreetech.com
# @File    : DataLoader.py
# @Software: PyCharm

import numpy as np
import os


def num2one_hot(data):
    data = data.astype(int)
    c = data.max()
    return np.eye(c)[data - 1]


def load_data(root, thd=30, gap=0.7):
    np.random.seed(1)
    files = os.listdir(root)
    datas = []
    for f in files:
        datas.extend(np.loadtxt(os.path.join(root, f)))
    datas = np.array(datas)
    m, n = datas.shape
    datas = datas[np.max(datas[:, :-1], axis=1) > thd, :]
    X = datas[:, :-1]
    X_real = X
    X = np.apply_along_axis(lambda x: x / np.max(np.abs(x)),
                            1,
                            X
                            )
    y = num2one_hot(datas[:, -1])
    z = list(zip(X_real, X, y))
    print(len(z))
    np.random.shuffle(z)
    X_real_, X_, y_ = zip(*z)
    gap = int(m * gap)
    X_train = np.array(X_[:gap])
    y_train = np.array(y_[:gap])
    X_real_train = np.array(X_real_[:gap])
    X_test = np.array(X_[gap:])
    y_test = np.array(y_[gap:])
    X_real_test = np.array(X_real_[gap:])
    return X_train, y_train, X_test, y_test, X_real_train, X_real_test


if __name__ == '__main__':
    x1, y1, x2, y2, x3, x4 = load_data('data', 30)
    print(x1.shape, y1.shape)
