import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable all debugging logs
import tensorflow as tf
import numpy as np
from comm_fn import minibatch, standard

# ToDo:
# Create nn class
# Create a cnn for nlp. Apply this convolution method https://arxiv.org/abs/1408.5882


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Dropout probability constants
P_KEEP_TRAIN=0.9
P_KEEP_TEST=1.0

p_keep_input = tf.placeholder(tf.float32)

x_no_drop = tf.placeholder(tf.float32, [None, 300])
x = tf.nn.dropout(x_no_drop, p_keep_input)
y_ = tf.placeholder(tf.float32, [None, 2])

W1 = init_weights([300, 100])
b1 = init_bias([100])
z1 = tf.matmul(x, W1) + b1
h1_no_drop = tf.nn.sigmoid(z1)
h1 = tf.nn.dropout(h1_no_drop, p_keep_input)

W2 = init_weights([100, 100])
b2 = init_bias([100])
z2 = tf.matmul(h1, W2) + b2
h2_no_drop = tf.sigmoid(z2)
h2 = tf.nn.dropout(h2_no_drop, p_keep_input)

W3 = init_weights([100, 2])
b3 = init_bias([2])
z3 = tf.matmul(h2, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z3, labels=y_))
train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
predict_op = tf.argmax(z3, 1)

correct_prediction = tf.equal(tf.argmax(z3,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
rand = np.random.RandomState(0)

X_train, X_test, y_train, y_test, X_ind = standard()

# shape (n_samples, 2)
y_train_hot = [[i, i-1] if i==1 else [i, i+1] for i in y_train]
y_train_hot = np.asarray(y_train_hot)
y_test_hot = [[i, i-1] if i==1 else [i, i+1] for i in y_test]
y_test_hot = np.asarray(y_test_hot)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(5000):
    X_batch, y_batch = minibatch(rand, X_train, y_train_hot, batch_size=1000)
    sess.run(train_op, {x_no_drop: X_batch, y_: y_batch, p_keep_input: P_KEEP_TRAIN})

    if i%1000 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: X_batch, y_: y_batch, p_keep_input:P_KEEP_TRAIN})
        curr_cost = sess.run(cost, {x_no_drop: X_batch, y_: y_batch, p_keep_input: P_KEEP_TRAIN})
        print("step %d, training accuracy %f, cost %f"%(i, train_accuracy, curr_cost))

print('---------------------')

# Reduce GPU ram usage by feeding data in batches
accuracy_list = []
for i in range(5):
    X_batch, y_batch = minibatch(rand, X_test, y_test_hot, batch_size=500)
    accuracy_list.append(sess.run(accuracy, {x_no_drop: X_batch, y_: y_batch, p_keep_input: P_KEEP_TEST}))

print(sum(accuracy_list)/len(accuracy_list))

