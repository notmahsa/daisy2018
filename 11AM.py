import numpy as np
import os
import csv
import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    pass
data_2009 = data_read('hackathon_dataset_2009.csv')
plot_promo_sold([8598, 22631, 102257], data_2009)