import numpy
import csv
def data_read(file_name):
    datafile = open(file_name, 'r')
    datareader = csv.reader(datafile, delimiter=';')
    data = []
    for row in datareader:
        data.append(row)
    return data
print data_read('hackathon_dataset_2009.csv')[0]
'''
import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()
'''
import matplotlib.pyplot as plt