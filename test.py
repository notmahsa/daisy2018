import numpy as np
import os
import csv


def data_read(filename):
    pass


def hypothesis(w, phi):
    h = np.matmul(w.transpose(), phi.transpose())
    return h.transpose()


def compute_cost(h, y):
    return (h.transpose()-y) ** 2


def normal_eq_regularized(phi, y, lam):
    eyes = np.eye(len(phi))
    norm = np.linalg.pinv(np.matmul(phi, phi.transpose()) + lam * eyes)
    norm = np.matmul(norm, phi)
    norm = np.matmul(norm.transpose(), y)
    return norm


def augment(x, m):
    phi = np.ones((len(x), 1))

    for i in range(1, m+1):
        temp = x ** i
        phi = np.concatenate((phi, temp), axis=1)

    return phi


def learn(m):
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            data_read(file)





if __name__ == "__main__":
    print()