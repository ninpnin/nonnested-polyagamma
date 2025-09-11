import numpy as np
from polyagamma import random_polyagamma
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import tqdm

y = np.array([1,0,1, 1, 1])
n = len(y)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def log_ll(a, b, y):
    ones = np.sum(y)
    zeros = len(y) - np.sum(y)

    pos = sigmoid(a * b)
    neg = sigmoid(- a * b)

    return np.log(pos) * ones + np.log(neg) * zeros

def log_post(a, b, y):
    ll = log_ll(a, b, y)
    #print(ll)
    return ll - 0.5 * a * a - 0.5 * b * b

def metropolis(r=1000):
    a_t, b_t = 0.0 ,0.0

    scale = 0.2

    rows = []
    for ix in tqdm.tqdm(range(r)):
        a_prop = a_t + np.random.randn() * scale
        b_prop = b_t + np.random.randn() * scale

        post_t = log_post(a_t, b_t, y)
        post_prop = log_post(a_prop, b_prop, y)

        ratio = np.exp(post_prop - post_t)
        #print(post_t, post_prop, ratio)

        if np.random.rand() < ratio:
            a_t = a_prop
            b_t = b_prop

        if ix % int(r / 1000) == 0:
            rows.append([a_t, b_t])

    df = pd.DataFrame(rows, columns=["a", "b"])
    return df

n_metro = 100000
df = metropolis(r=n_metro)
df = df.tail(int(n_metro*0.9))
print(df)

print(df.corr())

print(df.mean())
print(df.cov())

sns.scatterplot(df.iloc[::3, :], x="a", y="b")
plt.show()
plt.clf()


def pg_stuff(y):
    o = None
    a_t, b_t = 0.4, 0.5
    rows = []

    ones = np.sum(y)
    zeros = len(y) - np.sum(y)
    kappa = (ones - len(y)/2)

    for _ in range(2000):
        #def step1():
        o = random_polyagamma(z=a_t * b_t, size=len(y))
        osum = np.sum(o)
        #o_dot_y = np.dot(o, y)
        std = 1 / np.sqrt(b_t*b_t*osum+1)
        mean = b_t * kappa * (std **2)
        #mean = o_dot_y * b_t  / osum
        #std = 1/ np.sqrt(osum+1)

        a_t = np.random.randn() * std + mean

        #mean = o_dot_y * a_t  / osum
        std = 1/ np.sqrt(a_t*a_t*osum+1)
        mean = a_t * (ones - len(y)/2) * (std **2)

        b_t = np.random.randn() * std + mean
        rows.append([a_t, b_t])

    df = pd.DataFrame(rows, columns=["a", "b"])
    df = df.tail(900)
    return df
    #step1()
    #step1()

df = pg_stuff(y)
print(df.corr())

print(df.mean())
print(df.cov())

print(df)
sns.scatterplot(df, x="a", y="b")
plt.show()

