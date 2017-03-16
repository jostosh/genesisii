"""Adam for TensorFlow."""
import tensorflow as tf


class EveOptimizer(object):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, beta_3=0.999, epsilon=1e-8, decay=0.,
                 thl=0.1, thu=10., d_clip_lo=0.1, d_clip_hi=10.0, epsilon_feedback=1e-8):
        self.iterations = tf.Variable(0., dtype=tf.float32, trainable=False)
        self.lr = tf.Variable(lr, dtype=tf.float32, trainable=False)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.epsilon = epsilon
        self.epsilon_feedback = epsilon_feedback
        self.initial_decay = self.decay = decay
        self.thl = tf.constant(thl)
        self.thu = tf.constant(thu)
        self.d = tf.Variable(1., trainable=False)

        self.d_clip_lo = d_clip_lo
        self.d_clip_hi = d_clip_hi

        self.updates = []
        self.weights = []

    def get_updates(self, params, loss, constraints=[]):
        self.updates.append(tf.assign_add(self.iterations, 1))
        grads = tf.gradients(loss, params)

        #lr = self.lr
        t = self.iterations + 1
        not_first_iter = tf.greater(self.iterations, 1.)

        loss_prev =     tf.Variable(0., dtype=tf.float32, trainable=False)
        loss_hat_prev = tf.Variable(0., dtype=tf.float32, trainable=False)

        cond = tf.greater(loss, loss_prev)
        change_factor_lbound = tf.cond(cond, lambda: 1 + self.thl, lambda: 1/(1+self.thu))
        change_factor_ubound = tf.cond(cond, lambda: 1 + self.thu, lambda: 1/(1+self.thl))
        loss_change_factor = loss / loss_prev
        loss_change_factor = tf.maximum(loss_change_factor, change_factor_lbound)
        loss_change_factor = tf.minimum(loss_change_factor, change_factor_ubound)
        loss_hat = tf.cond(not_first_iter, lambda: loss_hat_prev * loss_change_factor, lambda: loss)

        d_den = tf.minimum(loss_hat, loss_hat_prev) #tf.cond(tf.greater(loss_hat, loss_prev), )
        d_t = (self.beta_3 * self.d) + (1. - self.beta_3) * tf.abs((loss_hat - loss_hat_prev) / (d_den + self.epsilon_feedback))
        d_t = tf.clip_by_value(
            tf.cond(not_first_iter, lambda: d_t, lambda: tf.constant(1.)),
            self.d_clip_lo,
            self.d_clip_hi
        )


        self.updates.append(tf.assign(self.d, d_t))

        lr_t = self.lr * (tf.sqrt(1 - tf.pow(self.beta_2, t)) / (1. - tf.pow(self.beta_1, t))) / \
               (1 + self.iterations * self.decay)

        shapes = [p.get_shape().as_list() for p in params]
        ms = [tf.Variable(tf.zeros(shape)) for shape in shapes]
        vs = [tf.Variable(tf.zeros(shape)) for shape in shapes]

        self.weights = [self.iterations, self.d, loss_prev] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * tf.square(g)
            p_t = p - lr_t * m_t / (tf.sqrt(v_t) * d_t + self.epsilon)

            self.updates.append(tf.assign(m, m_t))
            self.updates.append(tf.assign(v, v_t))

            new_p = p_t

            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(tf.assign(p, new_p))

        self.updates.append(tf.assign(loss_prev, loss))
        self.updates.append(tf.assign(loss_hat_prev, loss_hat))

        return self.updates
