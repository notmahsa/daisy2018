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


def read_all_files():
    file_num = 1

    # reading all the .csv files in the directory, saving their data into one 2D np array
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            if len(file.split('_')) == 3:
                if file_num == 1:
                    data = data_read(file)
                    file_num += 1
                else:
                    temp = data_read(file)
                    data = np.concatenate((data, temp), axis=0)

    return data


def hypothesis(w, phi):
    h = np.matmul(w.transpose(), phi.transpose())
    return h.transpose()


def cost(x, w, y, lam):
    m = y.shape[0]
    h = hypothesis(w, x)
    cost1 = np.power(np.subtract(h, y), 2)
    cost1 = np.sum(cost1) / (2 * m)
    cost2 = float(np.matmul(w.T, w) - w[0] ** 2) * lam * 0.5
    return cost1 + cost2


def normal_eq_regularized(phi, y, lam):
    eyes = np.eye(len(phi))
    norm = np.linalg.pinv(np.matmul(phi, phi.transpose()) + lam * eyes)
    norm = np.matmul(norm, phi)
    norm = np.matmul(norm.transpose(), y)
    return norm


def augment(x, pol_deg):
    phi = np.ones((x.shape[0], 1))

    for i in range(1, pol_deg + 1):
        temp = x ** i
        phi = np.concatenate((phi, temp), axis=1)

    return phi


def learn(pol_deg):
    data = read_all_files()
    m = data.shape[0]
    n = data.shape[1] - 1
    alpha = 0.1
    lam = 0.1
    B = int(m/50)
    num_epoch = 50
    num_iterations = int(num_epoch * (m / B))
    w = np.zeros(n).reshape(n, 1)

    # cleaning the data to set up the x and y matrices
    label_column = 5
    data_columns = np.arange(data.shape[1])
    data_columns = np.delete(data_columns, label_column, axis=0)

    for i in range(num_iterations):
        np.random.shuffle(data)
        x = data[:B, data_columns]
        y = data[:B, label_column].reshape(B, 1)

        # calculating the predictions on each data point and computing derivatives
        h = hypothesis(w, x)
        diff = h - y
        grad = np.matmul(x.T, diff)

        # computing the cost and prediction accuracy of current weights at every epoch
        if i % int(m / B) == 0:
            print('epoch', int(i/(m/B)))
            loss = cost(x, w, y, lam)
            #accurate = accuracy(x, w, y)
            #costs.append(loss)
            #acc.append(accurate)

        # updating weights
        w = w - (alpha / B) * grad + (lam / B) * w

    return w




if __name__ == "__main__":
    #x, y = read_all_files()
    learn(1)
