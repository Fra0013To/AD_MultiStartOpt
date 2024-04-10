import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from multistartopt.models import MultiStartOptimizationModel as MSOModel
from multistartopt.mso_nn import make_mso_model, find_levelset_mso_model

tf_dtype = tf.float32

verbose = True
y_level = 100.
EPOCHS = 5000


use_mso_class = True


def himmelblau(X):
    """
    y = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2
    :param X: matrix N-by-2, where rows are points in R^2
    :return y:
    """
    XX = tf.cast(X, dtype=tf_dtype)

    return (XX[:, 0]**2 + XX[:, 1] - 11)**2 + (XX[:, 0] + XX[:, 1]**2 - 7)**2


n = 2

pts_linspace = 100
N = pts_linspace ** 2
linspaces = (np.linspace(-7.5, 7.5, pts_linspace), np.linspace(-7.5, 7.5, pts_linspace))

mats = np.meshgrid(*linspaces)
X0 = mats[0].reshape((1, mats[0].size))
X0 = np.vstack([X0, mats[1].reshape((1, mats[1].size))])
X0 = X0.T

if use_mso_class:
    model = MSOModel(himmelblau, X0)
else:
    model = make_mso_model(n, himmelblau, X0)

adam_epsilon = 1e-7 / N
OPTIMIZER = tf.keras.optimizers.Adam(epsilon=adam_epsilon)

model.compile(optimizer=OPTIMIZER, loss='mse', metrics=['mae'])

if use_mso_class:
    t0 = time.time()
    Xlevel = model.find_level_set(y_level=y_level, epochs=EPOCHS, verbose=verbose)
else:
    t0 = time.time()
    Xlevel = find_levelset_mso_model(model, y_level, epochs=EPOCHS, verbose=verbose)
tfin = time.time()

deltat = tfin - t0
time_title = str(datetime.timedelta(seconds=deltat))

# -------------------- PLOTS ----------------------------------------------------
cont_linspaces = (np.linspace(-8, 8, 501), np.linspace(-8, 8, 501))
XX, YY = np.meshgrid(*cont_linspaces)
xxyy = np.hstack([XX.reshape(XX.size, 1), YY.reshape(YY.size, 1)])
zz = himmelblau(xxyy)
ZZ = zz.numpy().reshape(XX.shape)
extent = (-8., 8., -8., 8.)

fig1, ax1 = plt.subplots(1, 1)
pos1 = ax1.imshow(ZZ, origin='lower', alpha=0.75, extent=extent)
ax1.contour(XX, YY, ZZ, levels=[y_level], colors='lime')
ax1.scatter(Xlevel[:, 0], Xlevel[:, 1], marker='+', color='magenta', linewidths=0.5)
fig1.colorbar(pos1)
fig1.suptitle(f'HIMMELBLAU; N = {N}; y* = {y_level}; TIME: {time_title[:-3]}')

plt.show()