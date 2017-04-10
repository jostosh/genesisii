"""Adam for TensorFlow."""
import tensorflow as tf


class AdamOptimizer(object):
    """

	Added the comments 1, this is just for testing if the nbdiff works !

    """
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.):
        self.iterations = tf.Variable(0., dtype=tf.float32, trainable=False)
        self.lr = tf.Variable(lr, dtype=tf.float32, trainable=False)
        self.beta_1 = tf.constant(beta_1, dtype=tf.float32)
        self.beta_2 = tf.constant(beta_2, dtype=tf.float32)
        self.epsilon = epsilon
        self.initial_decay = self.decay = decay

        self.updates = []
        self.weights = []

    def get_updates(self, params, loss, constraints=[]):
        grads = tf.gradients(loss, params)
        self.updates.append(tf.assign_add(self.iterations, 1))

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * (tf.sqrt(1 - tf.pow(self.beta_2, t)) / (1. - tf.pow(self.beta_1, t)))

        shapes = [p.get_shape().as_list() for p in params]
        ms = [tf.Variable(tf.zeros(shape)) for shape in shapes]
        vs = [tf.Variable(tf.zeros(shape)) for shape in shapes]

        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * tf.square(g)
            p_t = p - lr_t * m_t / (tf.sqrt(v_t) + self.epsilon)

            self.updates.append(tf.assign(m, m_t))
            self.updates.append(tf.assign(v, v_t))

            new_p = p_t

            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(tf.assign(p, new_p))

        return self.updates
