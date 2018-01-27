import numpy as np
import csv
import os

def data_read(file_name):
    datafile = open(file_name, 'r')
    datareader = csv.reader(datafile, delimiter=';')
    data = []
    for row in datareader:
        data.append(row)
    return data


def read_all_files():
    file_num = 1

    # reading all the .csv files in the directory, saving their data into one 2D np array
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            if file_num == 1:
                data = data_read(file)
                file_num += 1
            else:
                temp = data_read(file)
                data = np.concatenate((data, temp), axis=0)

    # cleaning the data to set up the x,
    label_column = 5
    data_columns = np.arange(data.shape[1])
    data_columns = np.delete(data_columns, label_column, axis=1)
    x = data[:, data_columns]
    y = data[:, label_column]

    return x, y

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


def augment(x, pol_deg):
    phi = np.ones((x.shape[0], 1))

    for i in range(1, pol_deg + 1):
        temp = x ** i
        phi = np.concatenate((phi, temp), axis=1)

    return phi


def learn(pol_deg):
   x, y = read_all_files()
   phi = augment(x, pol_deg)
   lam = 0.1

   w = normal_eq_regularized(phi, y, lam)
   h = hypothesis(w, phi)

   return w, h








if __name__ == "__main__":
    data = data_read('hackathon_dataset_2009.csv')
    x = np.arange(6).reshape(2, -1)
    y = np.ones((2, 3))
    #print(np.concatenate((x, y), axis=0))

    a = np.arange(x.shape[1])
    b = np.delete(a, 1, axis=0)
    print(x[:, a])
    print(x[:, b])