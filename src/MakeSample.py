import numpy as np
from numpy.linalg import norm
from numpy.random import normal
import matplotlib.pyplot as plt
import os

A = 30
B = 20
RANGE = 20

TIMEFRAME = 100
NFEATURE = 10

epsilon = 1e-6


def ellipse():
    theta = np.linspace(0, 2*np.pi, TIMEFRAME+1)
    x = A * np.cos(theta)
    y = B * np.sin(theta)
    tan = np.arctan(- x / (y + epsilon))

    return np.vstack((x, y, tan))


def landmark():
    lmx = np.random.randint(-A-10, A+10, NFEATURE)
    lmy = np.random.randint(-B-10, B+10, NFEATURE)

    return np.vstack((lmx, lmy))


def main():
    x = ellipse()
    dx = np.diff(x)

    lm = landmark()
    lmdict = dict()

    t1 = 0
    t2 = 1

    with open("../data/sample2/input.txt", "w") as f_ob, open("../data/sample2/truedata.txt", "w") as f_tr:
        for t in range(TIMEFRAME):
            c = 0
            edge_tr = rotation_matrix(-x[2,t]) @ dx[:2,t]
            edge_ob = edge_tr + normal(0, 0.1, 2)
            tan_ob = dx[2,t] + normal(0, np.deg2rad(1.0))
            f_tr.write("EDGE_SE2 {} {} {} {} {} 0 0 0 0 0 0\n".format(t1, t2, edge_tr[0], edge_tr[1], dx[2,t]))
            f_ob.write("EDGE_SE2 {} {} {} {} {} 0 0 0 0 0 0\n".format(t1, t2, edge_ob[0], edge_ob[1], tan_ob))
            for j in range(NFEATURE):
                if norm(x[:2,t+1] - lm[:,j]) < RANGE:
                    if j not in lmdict.keys():
                        c += 1
                        lmdict[j] = t2 + c
                    edge_tr = rotation_matrix(-x[2,t+1]) @ (x[:2,t+1] - lm[:,j])
                    edge_ob = edge_tr + normal(0, 0.1, 2)
                    f_tr.write("EDGE_SE2_XY {} {} {} {} 0 0 0 0 0 0\n".format(t2, lmdict[j], edge_tr[0], edge_tr[1]))
                    f_ob.write("EDGE_SE2_XY {} {} {} {} 0 0 0 0 0 0\n".format(t2, lmdict[j], edge_ob[0], edge_ob[1]))
            t1 = t2
            t2 += 1 + c

    np.save("../data/sample2/xTrue.npy", x)
    np.save("../data/sample2/lmTrue.npy", lm)
    plt.plot(x[0,:], x[1,:])
    plt.scatter(lm[0,:], lm[1,:])
    plt.savefig("../data/sample2/truedata.png")
    plt.show()


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])


if __name__ == "__main__":
    main()
