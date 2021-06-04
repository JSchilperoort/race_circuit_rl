import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.enable_eager_execution()


class Critic(tf.keras.Model):
    """Critic model architecture"""

    def __init__(self, n_layers, n_units):
        super(Critic, self).__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        
        self.all_layers = list()

        # add hidden layers according to size 'n_layers' with 'n_units' per layer
        for _ in range(self.n_layers):
            self.all_layers.append(tf.keras.layers.Dense(self.n_units, activation='relu'))
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputstate, action):
        """feed forward step through the network, taking state and action as input and outputting values"""

        x = self.all_layers[0](tf.concat([inputstate, action], axis=1))
        for i in range(1, self.n_layers):
            x = self.all_layers[i](x)
        x = self.v(x)
        return x