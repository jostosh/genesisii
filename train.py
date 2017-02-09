from experiment_util import init_log_dir
from hyperparameters import HyperParameters, parse_cmd_args
import keras.backend as K
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from adam import AdamOptimizer
from eve import EveOptimizer
import numpy as np
from tensorboardutil import make_summary_from_python_var


def load_data(name):

    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    mnist = input_data.read_data_sets(data_dir,
                                      one_hot=True)

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    return mnist, x, y_

if __name__ == "__main__":
    hp = HyperParameters(parse_cmd_args())
    logdir = init_log_dir(hp)

    sess = tf.Session()
    K.set_session(sess)

    data, X, y_ = load_data(hp.data)

    prediction = hp.model(X, [-1, 28, 28, 1], 10)

    optimizer = (AdamOptimizer if hp.optimizer == 'adam' else EveOptimizer)(hp.lr)
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
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prediction)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    train_step = tf.group(*optimizer.get_updates(var_list, cross_entropy))

    if hp.optimizer == 'eve':
        tf.summary.scalar('d', optimizer.d)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(logdir + '/test')
    sess.run(tf.global_variables_initializer())


    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            xs, ys = data.train.next_batch(100)
        else:
            xs, ys = data.test.images, data.test.labels
        return {X: xs, y_: ys, K.learning_phase(): int(train)}


    for i in range(hp.max_steps):
        if i % 10 == 0:  # Record summaries and test-set accuracy
            accuracies = []
            for j in range(100):
                xs, ys = data.test.next_batch(100)
                summary, acc = sess.run([merged, accuracy], feed_dict={X: xs, y_:ys, K.learning_phase(): 0})
                accuracies.append(acc)
            print("Mean accuracy: {}".format(np.mean(accuracies)))
            test_writer.add_summary(make_summary_from_python_var('Accuracy', np.mean(accuracies, dtype='float')),  i)
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()
