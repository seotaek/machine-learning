# -*- coding: utf-8 -*-
"""
Written by Hanwool Jeong at Home
"""

import tensorflow as tf
import numpy as np

def neuron_layer(X, W, b, activation=None):
    z = tf.matmul(X, W)+b
    if activation is None:
        return z
    else:
        return activation(z)
    
def my_dMLP(X_flatten, W, b):
    hidden1 = neuron_layer(X_flatten,W[0], b[0], activation=tf.nn.sigmoid)
    hidden2 = neuron_layer(hidden1,  W[1], b[1], activation=tf.nn.sigmoid)
    # hidden1 = neuron_layer(X_flatten,W[0], b[0], activation=tf.nn.relu)
    # hidden2 = neuron_layer(hidden1,  W[1], b[1], activation=tf.nn.relu)
    """이거"""
    logits  = neuron_layer(hidden2,  W[2], b[2], activation=None)
    y_pred  = tf.nn.softmax(logits)
    return y_pred


# n_nodes are determined by W
n_inputs = np.array([28*28, 256, 128])
n_nodes  = np.array([256, 128, 10])
n_layer = 3
W, b = {}, {}
for layer in range(n_layer):
    stddev = 2 / np.sqrt(n_inputs[layer] + n_nodes[layer])
    W_init = tf.random.truncated_normal((n_inputs[layer], n_nodes[layer]), stddev = stddev)
    W[layer] = tf.Variable(W_init)
    b[layer] = tf.Variable(tf.zeros([n_nodes[layer]]))
    
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
nTrain = X_train.shape[0]
X_train_std,  X_test_std  = X_train/255.0,  X_test/255.0
X_train_std = X_train_std.astype("float32")

# convert class vectors to binary class matrices
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot  = tf.keras.utils.to_categorical(y_test, 10)

n_epoch = 40
batchSize = 200
nBatch = int(nTrain/batchSize)
# opt = tf.keras.optimizers.SGD(learning_rate=0.01)
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
"""이거"""

for epoch in range(n_epoch):
    idxShuffle = np.random.permutation(X_train.shape[0])
    for idxSet in range(nBatch):
        X_batch = X_train_std[idxShuffle[idxSet*batchSize:(idxSet+1)*batchSize], :]
        X_batch_tensor = tf.convert_to_tensor(X_batch.reshape(-1, 28*28))
        y_batch = y_train_onehot[idxShuffle[idxSet*batchSize:(idxSet+1)*batchSize], :]
        with tf.GradientTape() as tape:
            y_pred = my_dMLP(X_batch_tensor, W, b)
            """cross entropy"""
            # loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_batch, y_pred))
            loss = tf.reduce_mean(tf.keras.losses.MSE(y_batch, y_pred))
            """이거"""
        gradients = tape.gradient(loss, [W[2], W[1], W[0], b[2], b[1], b[0]])
        opt.apply_gradients(zip(gradients, [W[2], W[1], W[0], b[2], b[1], b[0]]))
        
    if epoch % 5 ==0:
        correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_batch, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)).numpy()
        print(accuracy)