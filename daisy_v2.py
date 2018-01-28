import tensorflow as tf
import numpy as np
import csv
import math
import matplotlib.pylab as plt

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
        data.append(temp[:5] + [0] + temp[ON_PROMO + 1:])
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
            row_num += 1
        elif temp[ITEM_NUMBER] == str(int(id)):
            data.append(
                list(map(float, temp[:5])) + [int(pred[row_num,0])] + list(map(float, temp[6:ON_PROMO])) + [
                    ((temp[ON_PROMO] != 'N') and b'Y') or b'N'] + list(map(float, temp[ON_PROMO + 1:])))
            row_num += 1
    with open("hackathon_result.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(data)

def split_test_item(data, list_of_items):
    out = []
    i = 0
    #list_of_items = np.unique(data[:,ITEM_NUMBER])
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

        learn(np.array(out), id)


def split_on_item(data, id):
    out = []
    i = 0
    #list_of_items = np.unique(data[:,ITEM_NUMBER])

    for item in data:
        if item[ITEM_NUMBER] == id:
            i += 1
            row = [
                math.sqrt(item[0] - DAY_ONE) / 4,
                item[1] / 70,
                item[2],
                item[4] / 10,
                0,
                item[6]
            ]
            out += [row]

    return np.array(out)


def augment(x, m):
    phi = np.copy(x)

    for i in range(1, m):
        temp = x ** (i+1)
        phi = np.concatenate((phi, temp), axis=1)

    return phi


def data_segmentation(data):
    m = data.shape[0]
    num_train = int(m * 0.8)
    num_valid = int(m * 0.2)

    train = data[:num_train, :]
    valid = data[num_train: num_train + num_valid, :]

    return train, valid


def learn(data, id):
    # reading the data into numpy 2D arrays
    train, valid = data_segmentation(data)
    test = data_result_read('hackathon_result.csv')
    test = split_on_item(test, id)

    label_column = 4
    data_columns = np.arange(train.shape[1])
    data_columns = np.delete(data_columns, label_column, axis=0)

    # setting up the training, validation, and test data sets
    x_train = train[:, data_columns]
    y_train = train[:, label_column].reshape(-1, 1)

    x_valid = valid[:, data_columns]
    y_valid = valid[:, label_column].reshape(-1, 1)

    x_test = test[:, data_columns]
    y_test = test[:, label_column].reshape(-1, 1)

    x_train = augment(x_train, 1)
    x_valid = augment(x_valid, 1)
    x_test = augment(x_test, 1)

    print(x_test.shape)

    # defining some parameter constants for the model
    lam = 1
    learning_rate = 0.1
    m = x_train.shape[0]
    n = x_train.shape[1]
    num_epoch = 50
    B = int(m/100)
    num_iterations = int(num_epoch * (m/B))

    # defining the model parameters
    X = tf.placeholder(tf.float32, shape=[None, n], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

    # initializing weights and biases (combined into one weights matrix) for the model
    W = tf.Variable(tf.truncated_normal(shape=[n, 1], stddev=0.5),name='weights')
    b = tf.Variable(np.random.randn(), name="bias")

    # constructing a linear model
    pred = tf.add(tf.matmul(X, W), b)

    # defining the cost as the mean squared error + regularization cost
    MSE_loss = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*m)
    reg_loss = (tf.reduce_sum(W * W)) * lam * 0.5
    cost = MSE_loss + reg_loss

    # defining the accuracy of prediction
    correct_preds = tf.equal(tf.round(pred), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    # training the model
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost)

    model = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(model)

        for i in range(num_iterations):
            # shuffling the data set for the stochastic gradient descent optimization
            randIdx = np.arange(len(x_train))
            np.random.shuffle(randIdx)
            randIdx = randIdx[:B]
            x_train = x_train[randIdx]
            y_train = y_train[randIdx]

            sess.run(train, feed_dict={X: x_train, Y: y_train})

            # calculating the error and accuracy every epoch iterations
            #if int(i % (m/B)) == 0:
                #print('---------- epoch:', int(i/(m/B)), '----------')

        predictions = sess.run(pred, feed_dict={X: x_test, Y: y_test})
        np.round(predictions, decimals=1)
        print(predictions.shape)
        complete_file(predictions, id)




if __name__ == "__main__":
    data_2009 = data_read('hackathon_dataset_2009.csv')
    data_2010 = data_read('hackathon_dataset_2010.csv')
    data_2011 = data_read('hackathon_dataset_2011.csv')
    data = np.concatenate((data_2009, data_2010, data_2011))

    data_result = data_result_read('hackathon_result.csv')

    result_items = np.unique((data_result[:, ITEM_NUMBER]))
    split_test_item(data, result_items)
