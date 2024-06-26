import tensorflow as tf
import numpy as np
from multistartopt.layers import FunctionLayer


def make_mso_model(function, starting_pts, dtype=tf.float32, name=None):
    n = starting_pts.shape[-1]

    I = tf.keras.layers.Input((n, ), dtype=dtype)

    W = starting_pts.astype(dtype.name)

    D = FunctionLayer(W.shape[0], function, dtype=dtype)(I)

    model = tf.keras.models.Model(inputs=I, outputs=D, name=name, dtype=dtype)

    model.set_weights([W.T])

    return model


def minimize_mso_model(mso_model, epochs=100, verbose=False, callbacks=None):
    Xtrain = np.ones((1, mso_model.input_shape[-1]))
    Ytrain = np.zeros((1, mso_model.layers[1].units))

    mso_model.fit(Xtrain, Ytrain, epochs=epochs, verbose=verbose, callbacks=callbacks)

    return mso_model.get_weights()[0].T


def find_levelset_mso_model(mso_model, y_level, epochs=100, verbose=False, callbacks=None):
    Xtrain = np.ones((1, mso_model.input_shape[-1]))
    Ytrain = y_level * np.ones((1, mso_model.layers[1].units))

    mso_model.fit(Xtrain, Ytrain, epochs=epochs, verbose=verbose, callbacks=callbacks)

    return mso_model.get_weights()[0].T


