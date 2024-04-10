import tensorflow as tf


class FunctionLayer(tf.keras.layers.Dense):
    """
    Given an activation function f, returns f applied to the layer's weights (transposed); i.e.:
    L(x) = f(W.T), for each input x
    """
    def __init__(self,
                 units,
                 activation,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **options):
        """
        Initialization method
        :param units: same of class Dense
        :param activation: same of class Dense
        :param use_bias: same of class Dense
        :param kernel_initializer: same of class Dense
        :param bias_initializer: same of class Dense
        :param kernel_regularizer: same of class Dense
        :param bias_regularizer: same of class Dense
        :param activity_regularizer: same of class Dense
        :param kernel_constraint: same of class Dense
        :param bias_constraint: same of class Dense
        :param options: same of class Dense
        """
        # ATTENTION: BIASES ARE NOT USED!
        super(FunctionLayer, self).__init__(units,
                                            activation=activation,
                                            use_bias=False,
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            activity_regularizer=activity_regularizer,
                                            kernel_constraint=kernel_constraint,
                                            **options)

    def get_config(self):

        config = super(FunctionLayer, self).get_config()

        return config

    def build(self, input_shape):
        # Kernel creation
        self.kernel = self.add_weight('kernel', shape=[int(input_shape[-1]), self.units],
                                      initializer=self.kernel_initializer
                                      )

        # ATTENTION: BIASES ARE NOT USED!
        # self.bias = self.add_weight('bias', shape=[self.units], initializer=self.bias_initializer)

    def call(self, input):
        # ATTENTION: HERE, THE INPUT ACTUALLY IS NOT USED!

        out_tensor = self.activation(tf.transpose(self.kernel))

        return out_tensor