from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, LSTM,Embedding
import tensorflow as tf
import numpy as np


def cnn_model(x, input_shape, nb_classes):
    """
    Convolutional Neural Network that also is used by
    Improving Stochastic Gradient Descent with Feedback
    :param X: tensor input for the model
    :param input_shape: shape of the input
    :param nb_classes: number of classes
    :return: the model
    """
    x = tf.reshape(x, input_shape)
    tf.summary.image('input', x, 10)

    x = Convolution2D(32, 3, 3, activation="relu", border_mode="same")(x)
    x = Convolution2D(32, 3, 3, activation="relu", border_mode="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Convolution2D(64, 3, 3, activation="relu", border_mode="same")(x)
    x = Convolution2D(64, 3, 3, activation="relu", border_mode="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = tf.reshape(x, [-1, np.prod(x.get_shape()[1:].as_list())]) #flatten
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation="softmax")(x)
    return x


def big_cnn_model(x, input_shape, nb_classes):
    x = tf.reshape(x, input_shape)
    x = Convolution2D(64, 3, 3, activation="relu", border_mode="same")(x)
    x = Convolution2D(64, 3, 3, activation="relu", border_mode="same")(x)
    x = Convolution2D(64, 3, 3, activation="relu", border_mode="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(128, 3, 3, activation="relu", border_mode="same")(x)
    x = Convolution2D(128, 3, 3, activation="relu", border_mode="same")(x)
    x = Convolution2D(128, 3, 3, activation="relu", border_mode="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(256, 3, 3, activation="relu", border_mode="same")(x)
    x = Convolution2D(256, 3, 3, activation="relu", border_mode="same")(x)
    x = Convolution2D(256, 3, 3, activation="relu", border_mode="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    x = tf.reshape(x, [-1, np.prod(x.get_shape()[1:].as_list())])
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation="softmax")(x)
    return x


def logistic_model(X, input_shape, nb_classes):
    """
    Logistic regression that also is used by
    Improving Stochastic Gradient Descent with Feedback
    :param X: tensor input for the model
    :param input_shape: shape of the input
    :param nb_classes: number of classes
    :return: the model
    """
    x = Dense(nb_classes, activation="softmax", input_shape=input_shape)(X)
    return x


def lstm(X, input_shape, nb_classes):
    """
    Basic LSTM that is also used in the examples of keras:
    https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py
    :param X: tensor input for the model
    :param input_shape: Number of input features that is used for the embedding
    :return: the model
    """

    x = Embedding(np.prod(input_shape[1:]), 128, dropout=0.2)(X)
    x = LSTM(128, dropout_W=0.2, dropout_U=0.2)(x)
    x = Dense(nb_classes, activation='linear')(x)
    return x


MODEL_FACTORIES = {
    "mnist": cnn_model,
    "cifar10": cnn_model,
    "cifar100": big_cnn_model,
    "oxflower": big_cnn_model,
    "logistic": logistic_model,
    "imdb" : lstm,
    "reuters": lstm
}
