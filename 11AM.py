import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import spline
DATE = 0
ITEM_NUMBER = 3
QUANTITY = 5
ON_PROMO = 6
PROMO = 6

def data_read(file_name):
    datafile = open(file_name, 'r')
    datareader = csv.reader(datafile, delimiter=';')
    data = []
    for [row] in datareader:
        temp = row.split(',')
        data.append(temp[:ON_PROMO] + temp[ON_PROMO + 1:])
    data = np.array(data[1:])
    return data.astype(np.float)


def hypothesis(w, phi):
    h = np.matmul(w.transpose(), phi.transpose())
    return h.transpose()


def compute_cost(h, y):
    return (h.transpose() - y) ** 2


def normal_eq_regularized(phi, y, lam):
    eyes = np.eye(len(phi))
    norm = np.linalg.pinv(np.matmul(phi, phi.transpose()) + lam * eyes)
    norm = np.matmul(norm, phi)
    norm = np.matmul(norm.transpose(), y)
    return norm


def augment(x, m):
    phi = np.ones((len(x), 1))

    for i in range(1, m + 1):
        temp = x ** i
        phi = np.concatenate((phi, temp), axis=1)

    return phi


def learn(m):
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            data_read(file)


def plot_promo_sold(items, data):

    for item in items:
        x = {}
        num = {}
        for entry in data:
            if entry[ITEM_NUMBER] == item:
                if entry[PROMO] not in x:
                    x[entry[PROMO]] = entry[QUANTITY]
                    num[entry[PROMO]] = 1
                else:
                    x[entry[PROMO]] += entry[QUANTITY]
                    num[entry[PROMO]] += 1
        for promo in x:
            x[promo] /= num[promo]
        import matplotlib
        matplotlib.pyplot.scatter(x.keys(), x.values())
    plt.xlabel("PROMO TYPE")
    plt.ylabel("AVERAGE # ITEMS SOLD")
    plt.show()

def plot_var_against_sold(items, data, xvar, xlabel, scatter = True):
    new = plt.figure()
    for item in items:
        x = {}
        num = {}
        for entry in data:
            if entry[ITEM_NUMBER] == item:
                if entry[xvar] not in x:
                    x[entry[xvar]] = entry[QUANTITY]
                    num[entry[xvar]] = 1
                else:
                    x[entry[xvar]] += entry[QUANTITY]
                    num[entry[xvar]] += 1
        for v in x:
            x[v] /= num[v]

        if scatter:
            plt.scatter(x.keys(), x.values())
        else:
            plt.plot(x.keys(), x.values())
    plt.xlabel(xlabel)
    plt.ylabel("AVERAGE # ITEMS SOLD")
    new.show()

def plot_var_against_promo(items, data, xvar, xlabel, scatter = True):
    new = plt.figure()
    for item in items:
        x = []
        y = []
        for entry in data:
            if entry[ITEM_NUMBER] == item:
                x += [entry[xvar]]
                y += [entry[PROMO]]

        if scatter:
            plt.scatter(x, y)
        else:
            plt.plot(x, x)

    plt.xlabel(xlabel)
    plt.ylabel("PROMO TYPES")
    new.show()

if __name__ == "__main__":
    pass
data_2009 = data_read('hackathon_dataset_2009.csv')
print len(np.unique(data_2009[:,0]))
plot_var_against_sold([8598, 22631, 102257, 263929, 423218], data_2009, PROMO, "Promo")
plot_var_against_sold([8598, 22631, 102257, 263929, 423218], data_2009, DATE, "Date")
plot_var_against_promo([8598, 22631, 102257, 263929, 423218], data_2009, DATE, "Date")
plt.show()