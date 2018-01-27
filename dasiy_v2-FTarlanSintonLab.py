import numpy as np
import csv
ITEM_NUMBER = 3
QUANTITY = 5
ON_PROMO = 6
PROMO = 6

def read_file(file_name):
    datafile = open(file_name, 'r')
    datareader = csv.reader(datafile, delimiter=';')
    data = []
    for [row] in datareader:
        temp = row.split(',')
        data.append(temp[:ON_PROMO] + temp[ON_PROMO + 1:])
    data = np.array(data[1:])
    return data.astype(np.float)


class Product(object):

    def __int__(self, date, store, department, item, price, quantity, on_promo, promo_type):
        self.data = [date, store, department, price, on_promo, promo_type]
        self.item = item
        self.quantity = quantity


def instantiate():
    filename = 'hackathon_dataset_2009.csv'
    data = read_file(filename)
    print(data[5, 3])

    items = []
    prods = []




if __name__ == "__main__":
    instantiate()