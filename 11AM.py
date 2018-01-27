import numpy
import csv
datafile = open('hackathon_dataset_2009.csv', 'r')
datareader = csv.reader(datafile, delimiter=';')
data = []
for row in datareader:
    data.append(row)
print data[0]
