import tensorflow as tf



class RMSPropOptimizer(object):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, beta_3=0.999, epsilon=1e-8, decay=0.,
                 thl=0.2, thu=5., d_clip_lo=0.1, d_clip_hi=10.0):
        self.iterations = tf.Variable(0., dtype=tf.float32, trainable=False)
        self.lr = tf.convert_to_tensor(tf.Variable(lr, dtype=tf.float32, trainable=False))
        self.beta_1 = beta_1
        self.epsilon = epsilon
        self.initial_decay = self.decay = decay
        self.thl = tf.constant(thl)
        self.thu = tf.constant(thu)
        self.d = tf.Variable(1., trainable=False)

        self.beta_2 = beta_2

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
        lr_t = lr * (tf.sqrt(1 - tf.pow(self.beta_1, t)))

        shapes = [p.get_shape().as_list() for p in params]
        vs = [tf.Variable(tf.zeros(shape)) for shape in shapes]
        self.weights = [self.iterations] + vs

        for p, g, v in zip(params, grads, vs):
            v_t = (self.beta_1 * v) + (1. - self.beta_1) * tf.square(g)
            p_t = p - lr_t * g / (tf.sqrt(v_t) + self.epsilon)

            self.updates.append(tf.assign(v, v_t))

            new_p = p_t

            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(tf.assign(p, new_p))

        return self.updates
