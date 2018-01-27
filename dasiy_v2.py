import tensorflow as tf
import numpy as np


lam = 0.1
learning_rate = 0.1

# defining the model parameters
X = tf.placeholder(tf.float32, shape=[None, n], name='X')
Y = tf.placeholder(tf.float32, shape=[None, n], name='Y')

# initializing weights and biases for the second netwrok
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# constructing a linear model
pred = tf.add(tf.multiply(X, W), b)

# defining the cost as the mean squared error + regularization cost
MSE_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
reg_loss = (tf.reduce_sum(tf.matmul(W, W))) * lam * 0.5
cost = MSE_loss + reg_loss

# training the model
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

model = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(model)

