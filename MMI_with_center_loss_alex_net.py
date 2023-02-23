# Parameters
LAMBDA = 0.01
CENTER_LOSS_ALPHA = 0.5
NUM_CLASSES = 6

# Import modules
import tensorflow as tf
import os
import numpy as np
import tflearn
import cv2
import time

slim = tf.contrib.slim

train_path = "../MMI_fold10_new/training_augment"
test_path = "../MMI_fold10_new/testing"
logs_path = "../logdir"

classes = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
image_size = 64
start_time = time.time()


def load_train(train_path, classes):
    data = []
    labels = []
    for fld in classes:
        index = classes.index(fld)
        cur_dir = os.path.join(train_path, fld)
        for image_file in sorted(os.listdir(cur_dir)):
            full_dir = os.path.join(cur_dir, image_file)
            print(full_dir)
            image = cv2.imread(full_dir)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            image = cv2.equalizeHist(image)
            image = np.array(image)
            image_flatten = image.flatten()
            data.append(image_flatten)

            # label = np.zeros(len(classes))
            # label[index] = 1.0
            labels.append(index)

    return data, labels


class DatasetSequence(object):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.batch_id = 0
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(self.data)

    def next(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finish epoch
            self._index_in_epoch += 1
            # shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.data = [self.data[e] for e in perm]
            self.labels = [self.labels[e] for e in perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.data[start:end], self.labels[start:end]


# Network Parameters
n_classes = len(classes)
data_train, labels_train = load_train(train_path, classes)
data_test, labels_test = load_train(test_path, classes)
data_train = np.array(data_train)
labels_train = np.array(labels_train)

trainset = DatasetSequence(data_train, labels_train)

data_test = np.array(data_test)
labels_test = np.array(labels_test)

testset = DatasetSequence(data_test, labels_test)

# Construct Network
with tf.name_scope('input'):
    input_images = tf.placeholder(tf.float32, shape=[None, 4096], name='input_images')
    labels = tf.placeholder(tf.int64, shape=(None), name='labels')
    keep_prob = tf.placeholder(tf.float32, name="dropout")

global_step = tf.Variable(0, trainable=False, name='global_step')


def get_center_loss(features, labels, alpha, num_classes):
    '''
    Arguments:
        features: Tensor, [batch_size, feature_length]
        labels: Tensor, [batch_size]
        alpha: 0-1
        num_classes
    Return:
        loss: Tensor, softmax loss, loss
        centers: Tensor
        centers_update_op: op
    '''

    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])

    centers_batch = tf.gather(centers, labels)
    print(features)
    # loss
    loss = tf.nn.l2_loss(features - centers_batch)

    # difference
    diff = centers_batch - features

    # mini-batch
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)

    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op


def inference(input_images,keep_prob):
    input_images1 = tf.reshape(input_images, [-1, image_size, image_size, 1])
    with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
        with slim.arg_scope([slim.max_pool2d], kernel_size=2):
            x = slim.conv2d(input_images1, num_outputs=64, scope='conv1_1')
            x = slim.conv2d(x, num_outputs=64, scope='conv1_2')
            x = slim.max_pool2d(x, scope='pool1')

            x = slim.conv2d(x, num_outputs=64, scope='conv2_1')
            x = slim.conv2d(x, num_outputs=64, scope='conv2_2')
            x = slim.max_pool2d(x, scope='pool2')

            x = slim.conv2d(x, num_outputs=128, scope='conv3_1')
            x = slim.conv2d(x, num_outputs=128, scope='conv3_2')
            x = slim.max_pool2d(x, scope='pool3')

            x = slim.flatten(x, scope='flatten')

            x = slim.fully_connected(x, num_outputs=1024, activation_fn=None, scope='fc1')
            x = tflearn.prelu(x)

            feature = slim.fully_connected(x, num_outputs=512, activation_fn=None, scope='fc2')
            x = tflearn.prelu(feature)

            x = slim.dropout(x, keep_prob, scope='dropout2')

            x = slim.fully_connected(x, num_outputs=n_classes, activation_fn=None, scope='fc3')

    return x, feature


def build_network(input_images, labels, ratio, keep_prob):
    logits, features = inference(input_images, keep_prob)

    with tf.name_scope('loss'):
        with tf.name_scope('center_loss'):
            center_loss, centers, centers_update_op = get_center_loss(features, labels, CENTER_LOSS_ALPHA, NUM_CLASSES)
        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        with tf.name_scope('total_loss'):
            total_loss = softmax_loss + ratio * center_loss

    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))

    with tf.name_scope('loss/'):
        tf.summary.scalar('CenterLoss', center_loss)
        tf.summary.scalar('SoftmaxLoss', softmax_loss)
        tf.summary.scalar('TotalLoss', total_loss)

    return logits, features, total_loss, accuracy, centers_update_op, center_loss, softmax_loss


logits, features, total_loss, accuracy, centers_update_op, center_loss, softmax_loss = build_network(input_images,
                                                                                                     labels,
                                                                                                     ratio=LAMBDA,
                                                                                                     keep_prob=keep_prob)

starter_learning_rate = 0.001
LEARNING_RATE_DECAY_FACTOR = 0.25
decay_steps = 6000
decayed_learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   LEARNING_RATE_DECAY_FACTOR,
                                                   staircase=True)
# Optimizer
optimizer = tf.train.RMSPropOptimizer(decayed_learning_rate)

with tf.control_dependencies([centers_update_op]):
    train_op = optimizer.minimize(total_loss, global_step=global_step)
predict = tf.argmax(logits, 1)
# Seesion and Summary

logs_path = "../logdir"
# config = tf.ConfigProto(device_count={'GPU': 0})
# sess = tf.Session(config=config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Initialize the variables
writer_test = tf.summary.FileWriter("../logdir/plot_test")
writer_train = tf.summary.FileWriter("../logdir/plot_train")
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("cost", total_loss)
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path, sess.graph)
# Train
mean_data = np.mean(trainset.data, axis=0)

step = sess.run(global_step)
while step <= 8000:
    batch_images, batch_labels = trainset.next(64)

    _, summary_str, train_acc, printed_total_loss, printed_cennter_loss, printed_softmax_loss = sess.run(
        [train_op, summary_op, accuracy, total_loss, center_loss, softmax_loss],
        feed_dict={input_images: batch_images - mean_data, labels: batch_labels, keep_prob: 0.7})
    print("Total_Loss: {:.6f}".format(printed_total_loss))
    print("Center_Loss: {:.6f}".format(printed_cennter_loss))
    print("Softmax_Loss: {:.6f}".format(printed_softmax_loss))
    step += 1
    writer_train.add_summary(summary_str, global_step=step)
    if step % 20 == 0:
        vali_image = testset.data - mean_data
        vali_acc, summary_str = sess.run([accuracy, summary_op],
                                         feed_dict={input_images: vali_image, labels: testset.labels, keep_prob: 1})
        writer_test.add_summary(summary_str, global_step=step)
        print(("step: {}, train_acc:{:.4f}, vali_acc:{:.4f}".format(step, train_acc, vali_acc)))
saver = tf.train.Saver()
save_path = saver.save(sess, "../model_alexnet_MMI/MMI_with_center_loss_alexnet_fold10.ckpt")

print(time.time() - start_time)
'''
# Visualize train_data

feat = sess.run(features, feed_dict={input_images: trainset.data })

# matplotlib inline
import matplotlib.pyplot as plt
labels = np.array(trainset.labels)

f = plt.figure(figsize=(16,9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
     '#ff00ff', '#990000']
for i in range(7):
    plt.plot(feat[labels == i, 0].flatten(), feat[labels == i, 1].flatten(), '.', c=c[i])


plt.legend(['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise' ])
plt.grid()
plt.show()

# Visualize test_data

feat = sess.run(features, feed_dict={input_images: testset.data})


# matplotlib inline
import matplotlib.pyplot as plt

labels = np.array(testset.labels)

f = plt.figure(figsize=(16, 9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
     '#ff00ff', '#990000']
for i in range(7):
    plt.plot(feat[labels == i, 0].flatten(), feat[labels == i, 1].flatten(), '.', c=c[i])

plt.legend(['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])
plt.grid()
plt.show()

'''
