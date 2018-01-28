import numpy as np
import os
import csv
import math
import matplotlib.pyplot as plt
from scipy.interpolate import spline
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


def data_result_read(file_name):
    datafile = open(file_name, 'r')
    datareader = csv.reader(datafile, delimiter=';')
    data = []
    for [row] in datareader:
        temp = row.split(',')
        data.append(temp[:5] + temp[6:ON_PROMO] + temp[ON_PROMO + 1:])
    data = np.array(data[1:])
    return data.astype(np.float)


def complete_file(pred, id):
    datafile = open('hackathon_result.csv', 'r')
    datareader = csv.reader(datafile, delimiter=';')
    data = []
    row_num = -1
    for [row] in datareader:
        temp = row.split(',')
        if row_num == -1:
            data.append(temp)
            row_num += 1
        elif row[ITEM_NUMBER] == str(id) and row_num > 0:
            data.append(
                temp[:5] + [pred[row_num]] + temp[6:ON_PROMO] + [
                    ((temp[ON_PROMO] != 'N') and 'Y') or 'N'] + temp[ON_PROMO + 1:])
            row_num += 1


    with open("hackathon_result.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(data)


def read_all_files():
    file_num = 1
    data = None
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


def plot_promo_price(items, data):
    new = plt.figure()
    for item in items:
        x = {}
        num = {}
        for entry in data:
            if entry[ITEM_NUMBER] == item:
                if entry[PROMO] not in x:
                    x[entry[PROMO]] = entry[PRICE]
                    num[entry[PROMO]] = 1
                else:
                    x[entry[PROMO]] += entry[PRICE]
                    num[entry[PROMO]] += 1
        for v in x:
            x[v] /= num[v]

        plt.scatter(x.keys(), x.values())
    plt.xlabel("PROMO")
    plt.ylabel("AVERAGE PRICE")
    new.show()

def split_on_item(data, list_of_items):
    out = []
    i = 0
    # list_of_items = np.unique(data[:,ITEM_NUMBER])
    for id in list_of_items:
        for item in data:
            if item[ITEM_NUMBER] == id:
                i += 1
                row = [
                    math.sqrt(item[0] - DAY_ONE) / 4,
                    item[1] / 70,
                    item[2],
                    item[4] / 10,
                    math.sqrt(math.sqrt(item[5])),
                    item[6]
                ]
                out += [row]
        # CALL FARHANG'S FUNCTION
        return np.array(out)


if __name__ == "__main__":
    # learn(1)
    pass

data_2009 = data_read('hackathon_dataset_2009.csv')
data_2010 = data_read('hackathon_dataset_2010.csv')
data_2011 = data_read('hackathon_dataset_2011.csv')
data = np.concatenate((data_2009, data_2010, data_2011))
data_result = data_result_read('hackathon_result.csv')

result_items = np.unique((data_result[:,ITEM_NUMBER]))
print split_on_item(data, result_items)
quit()

print(len(np.unique(data_2009[:,DATE])))
print(len(np.unique(data)))
print(len(np.unique(data_2009[:,PRICE])))
print(len(np.unique(data_2010[:,PRICE])))
print(len(np.unique(data_2011[:,PRICE])))
# plot_var_against_sold([8598, 22631, 102257, 263929, 423218], data_2009, PROMO, "Promo")
# plot_var_against_sold([8598, 22631, 102257, 263929, 423218], data_2009, DATE, "Date")
# plot_var_against_promo([8598, 22631, 102257, 263929, 423218], data_2009, DATE, "Date")
plot_var_against_sold([8598, 22631, 102257, 263929, 423218], data_2009, PRICE, "Price")
plot_promo_price([8598, 22631, 102257, 263929, 423218], data_2009)
plt.show()
