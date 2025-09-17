from trainerlog import get_logger
LOGGER = get_logger("benchmark", splitsec=True)
LOGGER.info("Load modules...")
import tensorflow as tf
from polyagamma import random_polyagamma
import numpy as np
import time
import polars as pl

def get_time():
    return time.time_ns() / 1_000_000

def get_delta(t0):
    t1 = get_time()
    delta = t1 - t0
    return delta, t1

K = 100
samples = 3000
D = 100

def get_v_omega_tf(X, omega):
    XT2 = tf.transpose(X, perm=[2,1,0])
    XTomega = (XT2 * omega)
    XTomega = tf.transpose(XTomega, perm=[2,1,0])

    V_inv = tf.linalg.matmul(X, XTomega, transpose_a=True)
    V_inv = tf.transpose(V_inv, perm=[0,2,1])
    return V_inv

def invert_v_omega_tf(V_inv):
    return tf.linalg.inv(V_inv)



omega = tf.constant(np.random.exponential(size=(samples, D)))

X = tf.constant(np.random.randn(D * K * samples).reshape((D, samples, K)))
LOGGER.info(f"omega: {omega.shape}")
LOGGER.info(f"X: {X.shape}")


time_data = {"delta": [], "phase": []}
Ls = [5, 10, 15]
for L in Ls:
    LOGGER.train(f"Do stacking for {L} vectors in the inversion")
    V_inv_prime = tf.random.normal(shape=(D // L, K * L, K * L))
    LOGGER.train(f"V_inv_prime {V_inv_prime.shape}")
    V_inv_prime = tf.linalg.matmul(V_inv_prime, V_inv_prime, transpose_a=True)
    V_inv_prime = V_inv_prime * 0.5 + tf.transpose(V_inv_prime, perm=[0,2,1]) * 0.5
    V_inv_prime += tf.eye(K * L, batch_shape=[D // L])
    LOGGER.train(f"V_inv_prime {V_inv_prime.shape}")
    TIME = get_time()

    for _ in  range(10):
        LOGGER.debug(f"Start")
        V_inv = get_v_omega_tf(X, omega)
        timedelta, TIME = get_delta(TIME)
        time_data["delta"] += [timedelta]
        time_data["phase"] += ["matmul"]
        LOGGER.debug(f"Matmul done")

        V = invert_v_omega_tf(V_inv)
        timedelta, TIME = get_delta(TIME)
        time_data["delta"] += [timedelta]
        time_data["phase"] += ["inversion"]
        LOGGER.debug(f"Inversion {V_inv.shape} done")

        L_chol = tf.linalg.cholesky(V)
        timedelta, TIME = get_delta(TIME)
        time_data["delta"] += [timedelta]
        time_data["phase"] += [f"cholesky"]

        V = invert_v_omega_tf(V_inv_prime)
        timedelta, TIME = get_delta(TIME)
        time_data["delta"] += [timedelta]
        time_data["phase"] += [f"inversion-correlated-{L}"]
        LOGGER.debug(f"Inversion {V_inv_prime.shape} done")

        L_chol = tf.linalg.cholesky(V)
        timedelta, TIME = get_delta(TIME)
        time_data["delta"] += [timedelta]
        time_data["phase"] += [f"cholesky-correlated-{L}"]
        #exit()

LOGGER.info(f"V_inv: {V_inv.shape}")

df = pl.DataFrame(time_data)
print(df)
print(df.group_by("phase").mean().sort("delta"))
print(df.group_by("phase").var())