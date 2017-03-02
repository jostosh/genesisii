from experiment_util import init_log_dir
from hyperparameters import HyperParameters, parse_cmd_args
import keras.backend as K
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from adam import AdamOptimizer
from eve import EveOptimizer
from genesis import GenesisOptimizer
import numpy as np
from tensorboardutil import make_summary_from_python_var
from keras.utils import np_utils
from keras.datasets import mnist, cifar10, cifar100, imdb, reuters
import tflearn.datasets.oxflower17 as oxflower17
import scipy.misc as misc
from keras.preprocessing import sequence
from keras.objectives import categorical_crossentropy

from tqdm import tqdm

def pre_process_image(hp):
    if hp.data == 'oxflower':
        X_data, y_data = DATASET_INFO[hp.data]["loader"](resize_pics=(64, 64))
        X_data = np.asarray([misc.imresize(im, (64, 64)) for im in X_data])
    else:
        (X_data, y_data), _ = DATASET_INFO[hp.data]["loader"]()
    if hp.n_samples == 0:
        hp.n_samples = X_data.shape[0]

    p = np.random.permutation(X_data.shape[0])
    X_data, y_data = X_data[p][:hp.n_samples], y_data[p][:hp.n_samples]
    X_data = X_data.astype("float32") / 255.
    if X_data.ndim == 3:
        X_data = X_data[:, :, :, np.newaxis]

    #y_data = np_utils.to_categorical(y_data, DATASET_INFO[hp.data]["nb_classes"])
    return X_data, np_utils.to_categorical(y_data,DATASET_INFO[hp.data]["nb_classes"])



def pre_process_text(hp):
    (X_train, y_train), _ = DATASET_INFO[hp.data]["loader"](nb_words=hp.n_vocab)
    if hp.n_samples > 0:
        X_train, y_train = X_train[:hp.n_samples], y_train[:hp.n_samples]
    X_train = sequence.pad_sequences(X_train, maxlen=hp.max_len)
    y_train = np.array(y_train)

    #y_train = np_utils.to_categorical(y_train, DATASET_INFO[hp.data]["nb_classes"])
    return X_train, y_train



DATASET_INFO = {
    "mnist": {"loader": mnist.load_data, "nb_classes": 10, "preprocess": pre_process_image},
    "cifar10": {"loader": cifar10.load_data, "nb_classes": 10, "preprocess": pre_process_image},
    "cifar100": {"loader": cifar100.load_data, "nb_classes": 100, "preprocess": pre_process_image},
    "oxflower": {"loader": oxflower17.load_data, "nb_classes": 17, "preprocess": pre_process_image},
    "imdb": {"loader": imdb.load_data, "nb_classes": 2, "preprocess": pre_process_text},
    "reuters": {"loader": reuters.load_data, "nb_classes": 46, "preprocess": pre_process_text}
}


def create_train_test(data, labels, cross_val_iter, k):
    test_length = data.shape[0] // k

    test_indices = range(cross_val_iter * test_length, (cross_val_iter + 1) * test_length)

    test_data = data[test_indices]
    test_labels = labels[test_indices]

    train_mask = np.ones(data.shape[0]).astype(np.bool)
    train_mask[test_indices] = False

    train_data = data[train_mask]
    train_labels = labels[train_mask]

    return train_data, train_labels, test_data, test_labels


if __name__ == "__main__":
    hp = HyperParameters(parse_cmd_args())
    logdir = init_log_dir(hp)

    print("Use data: {} and optimizer: {} and model: {}".format(hp.data, hp.optimizer, hp.model.__name__))

    sess = tf.Session()
    K.set_session(sess)

    X_data, y_data = DATASET_INFO[hp.data]["preprocess"](hp)
    X = tf.placeholder(tf.float32, (None,) + X_data.shape[1:])
    y_ = tf.placeholder(tf.float32, (None, DATASET_INFO[hp.data]['nb_classes'])) if hp.data not in ['imdb', 'reuters'] \
        else tf.placeholder(tf.int64, (None,))

    prediction = hp.model(X, (-1,) + X_data.shape[1:], DATASET_INFO[hp.data]['nb_classes'])

    optimizers = {
        'adam':     AdamOptimizer,
        'eve':      EveOptimizer,
        'genesis':  GenesisOptimizer
    }

    optimizer = optimizers[hp.optimizer](hp.lr)
    var_list = (tf.trainable_variables() + tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
    with tf.name_scope('cross_entropy'):
        # The raw formulation of cross-entropy,
        #
        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
        #                               reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.nn.softmax_cross_entropy_with_logits on the
        # raw outputs of the nn_layer above, and then average across
        # the batch.

        #diff = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=prediction)

        #diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prediction)

        #with tf.name_scope('total'):

        if hp.data in ['imdb', 'reuters']:
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=prediction))
        else:
            cross_entropy = tf.reduce_mean(categorical_crossentropy(y_, prediction))
            #cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    train_step = tf.group(*optimizer.get_updates(var_list, cross_entropy))

    if hp.optimizer == 'eve':
        tf.summary.scalar('d', optimizer.d)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_,1)) \
                if hp.data not in ['imdb', 'reuters'] else tf.equal(tf.argmax(prediction, 1), y_)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()

    for cross_iter in range(hp.cross_val):
        train_writer = tf.summary.FileWriter(logdir + '/cross{}/train'.format(cross_iter), sess.graph)
        test_writer = tf.summary.FileWriter(logdir + '/cross{}/test'.format(cross_iter))
        sess.run(tf.global_variables_initializer())

        X_train, y_train, X_test, y_test = create_train_test(X_data, y_data, cross_iter, hp.cross_val)

        # Train the model, and also write summaries.
        # Every 10th step, measure test-set accuracy, and write test summaries
        # All other steps, run train_step on training data, & add training summaries

        train_idx = 0
        test_idx = 0
        def feed_dict(train):
            global train_idx, test_idx
            """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
            if train:
                xs = X_train[train_idx:min(train_idx + hp.batch_size, X_train.shape[0])]
                ys = y_train[train_idx:min(train_idx + hp.batch_size, y_train.shape[0])]
                train_idx += hp.batch_size
                train_idx = 0 if train_idx >= X_train.shape[0] else train_idx
            else:
                xs = X_test[test_idx:min(test_idx + hp.batch_size, X_test.shape[0])]
                ys = y_test[test_idx:min(test_idx + hp.batch_size, y_test.shape[0])]
                test_idx += hp.batch_size
                test_idx = 0 if test_idx >= X_test.shape[0] else test_idx
           # if ys.ndim > 1: #Check if the dimensions are the same as the placeholder
           #     ys = ys[:, 0]
            return {X: xs, y_: ys, K.learning_phase(): int(train)}


        steps_per_epoch = int(np.ceil(X_train.shape[0] / hp.batch_size))

        i = 0
        for epoch in range(hp.epochs):
            accuracies = []
            for j in tqdm(range(int(np.ceil(X_test.shape[0] / hp.batch_size))), "Testing "):
                summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
                accuracies.append(acc)
            print("Mean accuracy: {}".format(np.mean(accuracies)))
            test_writer.add_summary(make_summary_from_python_var('Accuracy', np.mean(accuracies, dtype='float')), epoch)
            for j in tqdm(range(steps_per_epoch), "Training"):
                if i % 100 == 99:  # Record execution stats
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step],
                                          feed_dict=feed_dict(True),
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                else:  # Record a summary
                    sess.run([train_step], feed_dict=feed_dict(True))

                i += 1

        train_writer.close()
        test_writer.close()
