import cPickle
import numpy as np
import tensorflow as tf
from PIL import Image

def unpickle(file):
    fo = open(file, 'rb')
    dic = cPickle.load(fo)
    fo.close()
    return dic

train_data = []
train_label = []
test_data = []
test_label = []
train_data_temp = []
train_label_temp = []

batch_size = 250
epsilon = 0.001

initializer = tf.contrib.layers.xavier_initializer_conv2d()
filter_layer_1 = tf.Variable(initializer(shape=[3, 3, 3, 64]))
bias_layer_1 = tf.Variable(initializer(shape=[64]))

filter_layer_2 = tf.Variable(initializer(shape=[3, 3, 64, 128]))
bias_layer_2 = tf.Variable(initializer(shape=[128]))

filter_layer_3 = tf.Variable(initializer(shape=[3, 3, 128, 256]))
bias_layer_3 = tf.Variable(initializer(shape=[256]))

filter_layer_4 = tf.Variable(initializer(shape=[3, 3, 256, 256]))
bias_layer_4 = tf.Variable(initializer(shape=[256]))

weight_fc_1 = tf.Variable(initializer(shape=[4096, 1024]))
bias_fc_1 = tf.Variable(initializer(shape=[1024]))

weight_fc_2 = tf.Variable(initializer(shape=[1024, 1024]))
bias_fc_2 = tf.Variable(initializer(shape=[1024]))

weight_softmax = tf.Variable(initializer(shape=[1024, 10]))
bias_softmax = tf.Variable(initializer(shape=[10]))
#
# pre_activation_1 = tf.zeros([-1, 1024])
# activation_1 = tf.zeros([50000, 1024])
#
# pre_activation_2 = tf.zeros([50000, 1024])
# activation_2 = tf.zeros([50000, 1024])

# pre_activation_softmax = tf.zeros([50000, 10])
# activation_softmax = tf.zeros([50000, 10])

for j in range(5):
    d = unpickle('cifar-10-batches-py/data_batch_'+`j+1`)
    x = d['data']
    y = d['labels']

    train_data_temp.append(x)
    train_label_temp.append(y)

d=unpickle('cifar-10-batches-py/test_batch')
test_data = d['data']
test_label = d['labels']
test_label = np.reshape(test_label, [10000])
test_data = np.reshape(test_data, [10000, 32, 32, 3])

# print (type(test_data))

train_data = np.concatenate((train_data_temp[0],train_data_temp[1],train_data_temp[2],train_data_temp[3],train_data_temp[4]))
train_data = np.reshape(train_data, [-1, 32, 32, 3])
# train_data = tf.cast(train_data, tf.float32)
data = tf.placeholder(tf.float32, [None, 32, 32, 3])
label = tf.placeholder(tf.int64, [None])

train_label = np.concatenate((train_label_temp[0], train_label_temp[1], train_label_temp[2], train_label_temp[3], train_label_temp[4]))
# train_label = tf.reshape(train_label, [50000])
train_label_onehot = tf.one_hot(indices=tf.cast(train_label, tf.int32), depth=10)
print(train_label_onehot.shape)



con_layer_1 = tf.nn.conv2d(data, filter_layer_1, [1, 1, 1, 1], "SAME")
con_layer_1 = tf.nn.bias_add(con_layer_1, bias_layer_1)
con_layer_1 = tf.nn.relu(con_layer_1)

pool_layer_1 = tf.nn.max_pool(con_layer_1, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
print (pool_layer_1.shape)

con_layer_2 = tf.nn.conv2d(pool_layer_1, filter_layer_2, [1, 1, 1, 1], "SAME")
con_layer_2 = tf.nn.bias_add(con_layer_2, bias_layer_2)
con_layer_2 = tf.nn.relu(con_layer_2)
print (con_layer_2.shape)

pool_layer_2 = tf.nn.max_pool(con_layer_2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
print (pool_layer_2.shape)

con_layer_3 = tf.nn.conv2d(pool_layer_2, filter_layer_3, [1, 1, 1, 1], "SAME")
con_layer_3 = tf.nn.bias_add(con_layer_3, bias_layer_3)
con_layer_3 = tf.nn.relu(con_layer_3)
print (con_layer_3.shape)

con_layer_4 = tf.nn.conv2d(con_layer_3, filter_layer_4, [1, 1, 1, 1], "SAME")
con_layer_4 = tf.nn.bias_add(con_layer_4, bias_layer_4)
con_layer_4 = tf.nn.relu(con_layer_4)
# con_layer_4 = tf.reshape(con_layer_4, [-1, 4096])
print (con_layer_4.shape)

pool_layer_4 = tf.nn.max_pool(con_layer_4, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
print (pool_layer_4.shape)

pool_layer_4 = tf.reshape(pool_layer_4, [-1, 4096])

pre_activation_1 = tf.matmul(pool_layer_4, weight_fc_1) + bias_fc_1
activation_1 = tf.nn.relu(pre_activation_1)
print(activation_1.shape)

pre_activation_2 = tf.matmul(activation_1, weight_fc_2) + bias_fc_2
activation_2 = tf.nn.relu(pre_activation_2)
print(activation_1.shape)


batch_mean, batch_var = tf.nn.moments(activation_2,[0])
scale = tf.Variable(tf.ones([1024]))
beta = tf.Variable(tf.zeros([1024]))
batch_norm = tf.nn.batch_normalization(activation_2, batch_mean, batch_var, beta, scale, epsilon)


pre_activation_softmax = tf.matmul(batch_norm, weight_softmax) + bias_softmax
# activation_softmax = tf.nn.softmax(pre_activation_softmax)
# print(activation_softmax.shape)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pre_activation_softmax))
# print (loss.shape)

train_op = tf.train.AdamOptimizer(learning_rate=0.03).minimize(loss)
init_op = tf.global_variables_initializer()
# print (train_op.shape)
sess = tf.Session()
sess.run(init_op)

for i in range(0, 50000, batch_size):
    xyz = sess.run((train_op), feed_dict={data: train_data[i:i+batch_size], label: train_label[i:i+batch_size]})
    Loss = sess.run((loss), feed_dict={data: train_data[i:i+batch_size], label: train_label[i:i+batch_size]})
    print(Loss)
    print(i)

pred = tf.nn.softmax(pre_activation_softmax)
int_acc = tf.cast(tf.equal(tf.argmax(pred, 1), test_label[2000:8000]), tf.int32)
accuracy = tf.reduce_sum(int_acc)

acc = sess.run(accuracy, feed_dict={data: test_data[2000:8000], label: test_label[2000:8000]})
print(acc)