import numpy as np
import csv
import matplotlib.pylab as plt
import math

DATE = 0
ITEM_NUMBER = 3
PRICE = 4
QUANTITY = 5
ON_PROMO = 6
PROMO = 6
DAY_ONE = 20090101


def data_read(file_name):
    datafile = open(file_name, 'r')
    datareader = csv.reader(datafile, delimiter=';')
    data = []
    for [row] in datareader:
        temp = row.split(',')
        data.append(temp[:ON_PROMO] + temp[ON_PROMO + 1:])
    data = np.array(data[1:])
    return data.astype(np.float)


def split_on_item(data):
    out = []
    i = 0
    list_of_items = np.unique(data[:,ITEM_NUMBER])
    for id in list_of_items:
        for item in data:
            if item[ITEM_NUMBER] == id:
                i += 1
                row = [
                    math.sqrt(item[0] - DAY_ONE) / 4,
                    item[1] / 70,
                    item[2],
                    item[4] / 10,
                    item[5],
                    item[6]
                ]
                out += [row]
        # CALL FARHANG'S FUNCTION
        return np.array(out)


def predict(x, w):
    return np.matmul(x, w)


def cost(x, w, y, lam):
    m = y.shape[0]
    h = predict(x, w)
    cost1 = np.power(np.subtract(h, y), 2)
    cost1 = np.sum(cost1) / (2 * m)
    cost2 = float(np.matmul(w.T, w) - w[0] ** 2) * lam * 0.5
    return cost1 + cost2


def prediction_error(h, y):
    diff = h - y
    m = y.shape[0]
    return np.sum(diff ** 2) / (2*m)


def augment(x, m):
    one = np.ones((len(x), 1))
    phi = np.concatenate((one, x), axis=1)

    for i in range(1, m):
        temp = x ** (i+1)
        phi = np.concatenate((phi, temp), axis=1)

    return phi


def SGD(x, y):
    # assigning model constants
    learning_rate = 0.001
    lam = 0.1
    m = x.shape[0]
    n = x.shape[1]
    B = int(m/100)
    num_epoch = 30
    num_iter = int(num_epoch * (m / B))
    w = np.zeros(n).reshape(n, 1)

    costs = []

    # optimizing the weights num_iter times through stochastic gradient descent with batch size B
    for i in range(num_iter):
        # shuffling the data for the stochastic gradient descent iterations
        combined = np.concatenate((x, y), axis=1)
        np.random.shuffle(combined)
        x_shuf = combined[:B, :-1]
        y_shuf = combined[:B, -1].reshape(B, 1)

        # calculating the predictions on each data point and computing derivatives
        h = predict(x_shuf, w)
        loss = prediction_error(h, y_shuf)
        diff = h - y_shuf
        grad = np.matmul(x_shuf.T, diff)

        # computing the prediction error of current weights at every epoch iterations
        if i % int(m / B) == 0:
            print('epoch', int(i/(m/B)))
            costs.append(loss)

        # updating weights
        w = w - (learning_rate/B) * grad + (lam/B) * w

    return w, costs


if __name__ == "__main__":
    data = data_read('hackathon_dataset_2009.csv')
    data = split_on_item(data)

    label_column = 4
    data_columns = np.arange(data.shape[1])
    data_columns = np.delete(data_columns, label_column, axis=0)
    x = data[:, data_columns]
    y = data[:, label_column].reshape(-1, 1)
    x = augment(x, 1)

    print(x.shape)
    print(y.shape)

    w, err = SGD(x, y)

    epoch = np.arange(len(err))
    print(len(epoch))
    print(len(err))
    plt.plot(epoch, err)
    plt.xlabel('epoch')
    plt.ylabel('prediciton error')
