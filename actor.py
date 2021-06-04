import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.enable_eager_execution()


class Actor(tf.keras.Model):
    """Actor model architecture"""

    def __init__(self, no_action, n_layers, n_units):
        super(Actor, self).__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        
        self.all_layers = list()

        # add hidden layers according to size 'n_layers' with 'n_units' per layer
        for _ in range(self.n_layers):
            self.all_layers.append(tf.keras.layers.Dense(self.n_units, activation='relu'))
        self.mu = tf.keras.layers.Dense(no_action, activation='tanh')

    def call(self, state):
        """feed forward step through the network, taking state as input and outputting actions"""

        x = self.all_layers[0](state)
        for i in range(1, self.n_layers):
            x = self.all_layers[i](x)
        x = self.mu(x)
        return x
