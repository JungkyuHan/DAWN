import tensorflow as tf


class DenseBatchLayer(tf.keras.layers.Layer):
    def __init__(self, units, is_training, do_norm=True, regularizer_weight=None):
        super(DenseBatchLayer, self).__init__()
        self.units = units
        self.is_training = is_training
        self.do_norm = do_norm
        self.hidden_layer = None
        self.normalization_layer = None
        self.activation = tf.keras.layers.Activation(tf.nn.tanh)
        self.regularizer_weight = regularizer_weight

    def build(self, input_shape):
        truncated_normal_init = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
        zero_init = tf.initializers.Zeros()

        l2_regularizer = None
        if self.regularizer_weight is not None:
            l2_regularizer = tf.keras.regularizers.l2(self.regularizer_weight)

        self.hidden_layer = tf.keras.layers.Dense(units=self.units, activation=None, use_bias=True,
                                                  kernel_initializer=truncated_normal_init, bias_initializer=zero_init,
                                                  kernel_regularizer=l2_regularizer, bias_regularizer=l2_regularizer,
                                                  activity_regularizer=None, kernel_constraint=None,
                                                  bias_constraint=None)

        if self.do_norm:
            self.normalization_layer = tf.keras.layers.BatchNormalization(
                axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True,
                beta_initializer='zeros', gamma_initializer='ones',
                moving_mean_initializer='zeros',
                moving_variance_initializer='ones', beta_regularizer=None,
                gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                renorm=False, renorm_clipping=None, renorm_momentum=0.9, fused=None,
                trainable=self.is_training, virtual_batch_size=None, adjustment=None, name=None)

    def call(self, inputs):
        h_layer = self.hidden_layer(inputs)
        if self.do_norm:
            n_layer = self.normalization_layer(h_layer)
            layer_output = self.activation(n_layer)
        else:
            layer_output = self.activation(h_layer)
        return layer_output


class FCLayer(tf.keras.layers.Layer):
    def __init__(self, units, use_activation=True, regularizer_weight=None, AF="tanh"):
        super(FCLayer, self).__init__()
        self.units = units
        self.hidden_layer = None
        self.regularizer_weight = regularizer_weight
        self.use_activation = use_activation
        if AF == "sigmoid":
            self.activation = tf.keras.layers.Activation(tf.keras.activations.sigmoid)
        elif AF == "softmax":
            self.activation = tf.keras.layers.Activation(tf.keras.activations.softmax)
        else:
            self.activation = tf.keras.layers.Activation(tf.keras.activations.tanh)

    def build(self, input_shape):
        truncated_normal_init = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
        zero_init = tf.initializers.Zeros()

        l2_regularizer = None
        if self.regularizer_weight is not None:
            l2_regularizer = tf.keras.regularizers.l2(self.regularizer_weight)

        self.hidden_layer = tf.keras.layers.Dense(units=self.units, activation=None, use_bias=True,
                                                  kernel_initializer=truncated_normal_init, bias_initializer=zero_init,
                                                  kernel_regularizer=l2_regularizer, bias_regularizer=l2_regularizer,
                                                  activity_regularizer=None, kernel_constraint=None,
                                                  bias_constraint=None)

    def call(self, inputs):
        h_layer = self.hidden_layer(inputs)
        if self.use_activation:
            layer_output = self.activation(h_layer)
        else:
            layer_output = h_layer

        return layer_output


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr=0.001, decay=0.1, decay_step=10):
        super(CustomSchedule, self).__init__()
        self.lr = initial_lr
        self.decay = decay
        self.decay_step = decay_step
        self.epoch = 0

    def set_epoch(self, epoch=0):
        self.epoch = epoch
        if self.epoch % self.decay_step == 0:
            old_lr = self.lr
            self.lr = self.lr * self.decay
            print("epoch [%d]: lr changed from %f to %f" % (epoch, old_lr, self.lr))

    def __call__(self, step):
        return self.lr

    def get_config(self):
        base_config = super(CustomSchedule, self).get_config()
        base_config['lr'] = self.lr
        base_config['decay'] = self.decay
        base_config['decay_step'] = self.decay_step
        return base_config
