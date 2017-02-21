"""Adam for TensorFlow."""
import tensorflow as tf


class GenesisOptimizer(object):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, beta_3=0.999, epsilon=1e-8, decay=0.,
                 thl=0.25, thu=4.):
        self.iterations = tf.Variable(0., dtype=tf.float32, trainable=False)
        self.lr = tf.Variable(lr, dtype=tf.float32, trainable=False)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.epsilon = epsilon
        self.initial_decay = self.decay = decay
        self.thl = tf.constant(thl)
        self.thu = tf.constant(thu)
        self.d = tf.Variable(1., trainable=False)

        self.updates = []
        self.weights = []

    def get_updates(self, params, loss, constraints=[]):
        params = list(reversed(params))
        self.updates.append(tf.assign_add(self.iterations, 1))
        grads = tf.gradients(loss, params)
        #errors = tf.gradients(loss, activations)

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        not_first_iter = tf.greater(self.iterations, 1.)
        '''

        loss_prev = tf.Variable(0., dtype=tf.float32, trainable=False)

        cond = tf.greater(loss, loss_prev)
        ch_fact_lbound = tf.cond(cond, lambda: 1 + self.thl, lambda: 1/(1+self.thu))
        ch_fact_ubound = tf.cond(cond, lambda: 1 + self.thu, lambda: 1/(1+self.thl))
        loss_ch_fact = loss / loss_prev
        loss_ch_fact = tf.maximum(loss_ch_fact, ch_fact_lbound)
        loss_ch_fact = tf.minimum(loss_ch_fact, ch_fact_ubound)
        loss_hat = tf.cond(not_first_iter, lambda: loss_prev * loss_ch_fact, lambda: loss)

        d_den = tf.minimum(loss_hat, loss_prev) #tf.cond(tf.greater(loss_hat, loss_prev), )
        d_t = (self.beta_3 * self.d) + (1. - self.beta_3) * tf.abs((loss_hat - loss_prev) / d_den)
        d_t = tf.cond(not_first_iter, lambda: d_t, lambda: tf.constant(1.))
        self.updates.append(tf.assign(self.d, d_t))
        '''

        lr_t = lr * (tf.sqrt(1 - tf.pow(self.beta_2, t)) / (1. - tf.pow(self.beta_1, t)))

        shapes = [p.get_shape().as_list() for p in params]
        ms = [tf.Variable(tf.zeros(shape)) for shape in shapes]
        vs = [tf.Variable(tf.zeros(shape)) for shape in shapes]

        self.weights = [self.iterations] + ms + vs

        loss_next_layer = loss
        weights_last = None

        for p, g, m, v in zip(params, grads, ms, vs):

            if p.name in ["dense_2_W:0", "dense_1_W:0"]:
                loss_per_neuron = (tf.ones(p.get_shape().as_list()[1:], dtype=tf.float32) * loss_next_layer if '2' in p.name \
                    else tf.abs(tf.squeeze(tf.matmul(weights_last, tf.expand_dims(loss_next_layer, 1)))))
                #print(loss_per_neuron)
                loss_per_neuron_prev = tf.Variable(tf.ones(p.get_shape().as_list()[1:]), trainable=False, dtype=tf.float32)

                d = tf.Variable(tf.ones(p.get_shape().as_list()[1:]), trainable=False, dtype=tf.float32)

                cond = tf.cast(tf.greater(loss_per_neuron, loss_per_neuron_prev), dtype=tf.float32)

                ch_fact_lbound = cond * (1 + self.thl) + (1 - cond) / (1 + self.thu) #tf.cond(cond, lambda: 1 + self.thl, lambda: 1 / (1 + self.thu))
                ch_fact_ubound = cond * (1 + self.thu) + (1 - cond) / (1 + self.thl) #tf.cond(cond, lambda: 1 + self.thu, lambda: 1 / (1 + self.thl))
                loss_ch_fact = loss_per_neuron / loss_per_neuron_prev
                loss_ch_fact = tf.maximum(loss_ch_fact, ch_fact_lbound)
                loss_ch_fact = tf.minimum(loss_ch_fact, ch_fact_ubound)
                loss_hat = tf.cond(not_first_iter, lambda: loss_per_neuron_prev * loss_ch_fact,
                                   lambda: loss_per_neuron_prev)

                d_den = tf.minimum(loss_hat, loss_per_neuron_prev)  # tf.cond(tf.greater(loss_hat, loss_prev), )
                d_t = (self.beta_3 * d) + (1. - self.beta_3) * tf.abs((loss_hat - loss_per_neuron_prev) / d_den)
                d_t = tf.cond(not_first_iter, lambda: d_t, lambda: tf.ones_like(d_t, dtype=tf.float32))
                self.updates.append(tf.assign(d, d_t))

                m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * tf.square(g)

                #print(v_t.get_shape(), tf.matmul(tf.sqrt(v_t), tf.diag(d_t)).get_shape())
                p_t = p - lr_t * m_t / (tf.matmul(tf.sqrt(v_t), tf.diag(d_t)) + self.epsilon)
                self.updates.append(tf.assign(loss_per_neuron_prev, loss_hat))
                weights_last = p

                loss_next_layer = loss_per_neuron
            else:
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

        #for u in self.updates:
         #   print(u)


        return self.updates
