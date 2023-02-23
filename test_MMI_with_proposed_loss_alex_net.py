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

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
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
data_test, labels_index_test = load_train(test_path, classes)
data_train = np.array(data_train)
labels_train = np.array(labels_train)

trainset = DatasetSequence(data_train, labels_train)
print(len(trainset.data))
print(trainset._num_examples)


# Construct Network
with tf.name_scope('input'):
    input_images = tf.placeholder(tf.float32, shape=[None, 4096], name='input_images')
    labels = tf.placeholder(tf.int64, shape=(None), name='labels')
    keep_prob = tf.placeholder(tf.float32, name="dropout")

global_step = tf.Variable(0, trainable=False, name='global_step')
labels_center_1 = []
labels_center_2 = []


for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if (i==j) or (j<i):
            continue
        else:
            print([i,j])
            labels_center_1.append(i)
            labels_center_2.append(j)

labels_center_1 = tf.convert_to_tensor(labels_center_1)
labels_center_2 = tf.convert_to_tensor(labels_center_2)
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
    num_features = tf.shape(features)[0]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)


    labels_0 = tf.fill([num_features,], 0)
    labels_0 = tf.reshape(labels_0, [-1])
    centers_batch_0 = tf.gather(centers, labels_0)
    loss_0 = tf.nn.l2_loss(features - centers_batch_0)

    labels_1 = tf.fill([num_features, ], 1)
    labels_1 = tf.reshape(labels_1, [-1])
    centers_batch_1 = tf.gather(centers, labels_1)
    loss_1 = tf.nn.l2_loss(features - centers_batch_1)

    labels_2 = tf.fill([num_features, ], 2)
    labels_2 = tf.reshape(labels_2, [-1])
    centers_batch_2 = tf.gather(centers, labels_2)
    loss_2 = tf.nn.l2_loss(features - centers_batch_2)

    labels_3 = tf.fill([num_features, ], 3)
    labels_3 = tf.reshape(labels_3, [-1])
    centers_batch_3 = tf.gather(centers, labels_3)
    loss_3 = tf.nn.l2_loss(features - centers_batch_3)

    labels_4 = tf.fill([num_features, ], 4)
    labels_4 = tf.reshape(labels_4, [-1])
    centers_batch_4 = tf.gather(centers, labels_4)
    loss_4 = tf.nn.l2_loss(features - centers_batch_4)

    labels_5 = tf.fill([num_features, ], 5)
    labels_5 = tf.reshape(labels_5, [-1])
    centers_batch_5 = tf.gather(centers, labels_5)
    loss_5 = tf.nn.l2_loss(features - centers_batch_5)

    labels_6 = tf.fill([num_features, ], 6)
    labels_6 = tf.reshape(labels_6, [-1])
    centers_batch_6 = tf.gather(centers, labels_6)
    loss_6 = tf.nn.l2_loss(features - centers_batch_6)

    total_loss = tf.add(loss_0, loss_1)
    total_loss = tf.add(total_loss, loss_2)
    total_loss = tf.add(total_loss, loss_3)
    total_loss = tf.add(total_loss, loss_4)
    total_loss = tf.add(total_loss, loss_5)
    total_loss = tf.add(total_loss, loss_6)


    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)
    # loss
    intra_loss = tf.nn.l2_loss(features - centers_batch)

    inter_loss = tf.subtract(total_loss, intra_loss)
    epsilon1 = tf.constant(10e-3,dtype=tf.float32)
    inter_counter_loss = tf.divide(tf.constant(1.,dtype=tf.float32), tf.add(epsilon1, inter_loss))

    # calculate distance between center points
    centers_point_batch_1 =  tf.gather(centers, labels_center_1)
    centers_point_batch_2 = tf.gather(centers, labels_center_2)
    pair_center_loss = tf.nn.l2_loss(centers_point_batch_1 - centers_point_batch_2)

    epsilon2 = tf.constant(10e-3,dtype=tf.float32)
    counter_pair_center_loss = tf.divide(tf.constant(1.,dtype=tf.float32), tf.add(epsilon2, pair_center_loss))


    # difference
    diff = centers_batch - features
    # differences update from center point

    # mini-batch
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1,1])

    diff = diff/ tf.cast((1+appear_times), tf.float32)
    diff = alpha* diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)


    return  intra_loss, inter_counter_loss, counter_pair_center_loss,inter_loss, pair_center_loss,  centers, centers_update_op

def inference(input_images,keep_prob):#with slim.arg_scope([slim.conv2d],padding='SAME',weights_regularizer=slim.l2_regularizer(0.001)):
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

            x = slim.dropout(x,keep_prob, scope='dropout2')

            x = slim.fully_connected(x, num_outputs=n_classes, activation_fn=None, scope='fc3')

    return x, feature

def build_network(input_images, labels, ratio, keep_prob):
    logits, features = inference(input_images,keep_prob)

    with tf.name_scope('loss'):
        with tf.name_scope('center_loss'):
            intra_loss, inter_counter_loss, counter_pair_center_loss, inter_loss, pair_center_loss, centers, centers_update_op = get_center_loss(features, labels, CENTER_LOSS_ALPHA, NUM_CLASSES)
        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        with tf.name_scope('total_loss'):
            total_loss = softmax_loss + ratio * (intra_loss + 0.5* inter_counter_loss + 0.5 * counter_pair_center_loss)

    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))

    with tf.name_scope('loss/'):
        tf.summary.scalar('Softmax_Loss', softmax_loss)
        tf.summary.scalar('Intra_Loss', intra_loss)
        tf.summary.scalar('Inter_Counter_Loss', inter_counter_loss)
        tf.summary.scalar('Counter_Pair_Center_Loss', counter_pair_center_loss)
        tf.summary.scalar('Inter_Loss', inter_loss)
        tf.summary.scalar('Pair_Center_Loss', pair_center_loss)
        tf.summary.scalar('TotalLoss', total_loss)

    return logits, features, total_loss,softmax_loss, intra_loss, inter_loss, pair_center_loss, accuracy, centers_update_op


logits, features, total_loss, softmax_loss,intra_loss,inter_loss , pair_center_loss, accuracy, centers_update_op = build_network(input_images,labels, ratio=LAMBDA, keep_prob=keep_prob)

starter_learning_rate = 0.001
LEARNING_RATE_DECAY_FACTOR = 0.25
decay_steps = 8000
decayed_learning_rate=tf.train.exponential_decay(starter_learning_rate,
                                                     global_step,
                                                     decay_steps,
                                                     LEARNING_RATE_DECAY_FACTOR,
                                                     staircase=True)
# Optimizer
optimizer = tf.train.RMSPropOptimizer(decayed_learning_rate)

with tf.control_dependencies([centers_update_op]):
    train_op = optimizer.minimize(total_loss, global_step=global_step)
predict = tf.argmax(logits, 1)
data_test = np.array(data_test)
labels_index_test = np.array(labels_index_test)

num_test = len(labels_index_test)
print(num_test)
predict_labels = np.zeros(num_test)
batch_size1=1
# Seesion and Summary
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# Train
mean_data = np.mean(trainset.data, axis=0)
saver = tf.train.Saver()
save_path = saver.restore(sess, "../model_alexnet_MMI/MMI_with_proposed_loss_alexnet_fold10.ckpt")
vali_image = data_test - mean_data
for i in range(0,num_test//batch_size1):
    predict_labels[i*batch_size1 : (i+1)*batch_size1]= sess.run(predict, feed_dict={input_images: vali_image[i*batch_size1 : (i+1)*batch_size1], keep_prob:1})
print(predict_labels)
print(labels_index_test)

print(accuracy_score(labels_index_test, predict_labels))

cnf_matrix = confusion_matrix(labels_index_test, predict_labels)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(cm[i, j]*100)/100.0,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=classes, normalize=False, title='Confusion Matrix for Test Dataset')
plt.show()
