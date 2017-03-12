"""Adam for TensorFlow."""
import tensorflow as tf



class RMSEveOptimizer(object):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, beta_3=0.999, epsilon=1e-4, decay=0.,
                 thl=0.2, thu=5., d_clip_lo=0.1, d_clip_hi=10.0):
        self.iterations = tf.Variable(0., dtype=tf.float32, trainable=False)
        self.lr = tf.convert_to_tensor(tf.Variable(lr, dtype=tf.float32, trainable=False))
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.epsilon = epsilon
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

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        not_first_iter = tf.greater(self.iterations, 1.)

        loss_prev = tf.Variable(0., dtype=tf.float32, trainable=False)
        loss_hat_prev = tf.Variable(0., dtype=tf.float32, trainable=False)

        cond = tf.greater(loss, loss_prev)
        ch_fact_lbound = tf.cond(cond, lambda: 1 + self.thl, lambda: 1/(1+self.thu))
        ch_fact_ubound = tf.cond(cond, lambda: 1 + self.thu, lambda: 1/(1+self.thl))
        loss_ch_fact = loss / loss_prev
        loss_ch_fact = tf.maximum(loss_ch_fact, ch_fact_lbound)
        loss_ch_fact = tf.minimum(loss_ch_fact, ch_fact_ubound)
        loss_hat = tf.cond(not_first_iter, lambda: loss_hat_prev * loss_ch_fact, lambda: loss)

        d_den = tf.minimum(loss_hat, loss_hat_prev) #tf.cond(tf.greater(loss_hat, loss_prev), )
        d_t = (self.beta_3 * self.d) + (1. - self.beta_3) * tf.abs((loss_hat - loss_hat_prev) / d_den)
        d_t = tf.clip_by_value(
            tf.cond(not_first_iter, lambda: d_t, lambda: tf.constant(1.)),
            self.d_clip_lo,
            self.d_clip_hi
        )
        self.updates.append(tf.assign(self.d, d_t))

        lr_t = lr * tf.sqrt(1 - tf.pow(self.beta_1, t))

        shapes = [p.get_shape().as_list() for p in params]
        vs = [tf.Variable(tf.zeros(shape)) for shape in shapes]

        self.weights = [self.iterations, self.d, loss_prev] + vs

        for p, g, v in zip(params, grads, vs):
            v_t = (self.beta_1 * v) + (1. - self.beta_1) * tf.square(g)
            p_t = p - lr_t * g / (tf.sqrt(v_t) * d_t + self.epsilon)

            self.updates.append(tf.assign(v, v_t))

            new_p = p_t

            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(tf.assign(p, new_p))

        self.updates.append(tf.assign(loss_prev, loss))
        self.updates.append(tf.assign(loss_hat_prev, loss_hat))

        return self.updates
