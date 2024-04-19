import tensorflow as tf
import numpy as np
from multistartopt.layers import FunctionLayer


class MultiStartOptimizationModel(tf.keras.models.Model):
    def __init__(self,
                 function,
                 starting_pts,
                 dtype=tf.float32,
                 **kwargs
                 ):
        """
        Model initialization.
        :param function: tensorflow function (better if vectorized for batches of inputs)
        :param starting_pts: numpy array N-by-n (N points in R^n)
        :param kwargs:
        """
        super().__init__(dtype=dtype, **kwargs)

        self._tf_dtype = dtype
        self._starting_pts = tf.cast(starting_pts, dtype=self._tf_dtype)
        self._N, self._n = starting_pts.shape
        self._function = function

        self._func_layer = FunctionLayer(units=self._N, activation=self._function, dtype=self._tf_dtype)

        self._fake_input = np.ones((1, self._n))
        self.call(self._fake_input)
        self.set_weights([self._starting_pts.numpy().T])

    def get_config(self):

        config = super().get_config()

        config['starting_pts'] = self._starting_pts
        config['N'] = self._N
        config['n'] = self._n
        config['function'] = self._function
        config['func_layer'] = self._func_layer
        config['fake_input'] = self._fake_input

        return config

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, dtype=self._tf_dtype)
        tens = self._func_layer(inputs)

        return tens

    def minimize(self,
                 epochs=100,
                 verbose=False,
                 callbacks=None,
                 initial_epoch=0,
                 ):
        self.fit(
            x=self._fake_input,
            y=np.zeros((1, self._N)),
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            initial_epoch=initial_epoch,
        )

        return self.get_weights()[0].T

    def find_level_set(self,
                       y_level,
                       epochs=100,
                       verbose=False,
                       callbacks=None,
                       initial_epoch=0,
                       ):
        self.fit(
            x=self._fake_input,
            y=y_level * np.ones((1, self._N)),
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            initial_epoch=initial_epoch,
        )

        return self.get_weights()[0].T

