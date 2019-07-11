import numpy as np
from numpy.linalg import norm
from numpy.random import normal
import matplotlib.pyplot as plt
import os

A = 30
B = 20
RANGE = 20

TIMEFRAME = 1000
NFEATURE = 100

epsilon = 1e-6


def ellipse():
    theta = np.linspace(0, 2*np.pi, TIMEFRAME+1)
    x = A * np.cos(theta)
    y = B * np.sin(theta)
    tan = np.arctan2(B**2*x, -A**2*y)
    tan[0] = 0

    return np.vstack((x, y, tan))


def line():
    x = np.zeros(TIMEFRAME+1)
    y = np.linspace(0, 50, TIMEFRAME+1)
    tan = np.arctan2(B ** 2 * x, -A ** 2 * y)
    tan[0] = 0

    return np.vstack((x, y, tan))


def landmark():
    # lmx = np.random.randint(-A-10, A+10, NFEATURE)
    # lmy = np.random.randint(-B-10, B+10, NFEATURE)

    lmx = np.random.randint(A-10, A+10, NFEATURE)*np.cos(np.random.randint(0,2*np.pi,NFEATURE))
    lmy = np.random.randint(B-10, B+10, NFEATURE)*np.cos(np.random.randint(0,2*np.pi,NFEATURE))

    # lmx = np.random.randint(-10, 10, NFEATURE)
    # lmy = np.random.randint(-10, 50+10, NFEATURE)

    return np.vstack((lmx, lmy))


def main():

    dirname = "../data/sample{}".format(len(os.listdir("../data/")))
    os.mkdir(dirname)
    inputdata = os.path.join(dirname, "input.txt")
    truedata = os.path.join(dirname, "truedata.txt")
    xnpy = os.path.join(dirname, "xTrue.npy")
    lmnpy = os.path.join(dirname, "lmTrue.npy")
    fig = os.path.join(dirname, "truedata.png")

    x = ellipse()
    # x = line()
    dx = np.diff(x)
    dx[2,:] = (dx[2,:] + np.pi)%(2*np.pi) - np.pi

    lm = landmark()
    lmdict = dict()

    t1 = 0
    t2 = 1

    with open(inputdata, "w") as f_ob, open(truedata, "w") as f_tr:
        for t in range(TIMEFRAME):
            c = 0
            edge_tr = rotation_matrix(-x[2,t]) @ dx[:2,t]
            edge_ob = edge_tr + normal(0, 0.3, 2)
            tan_ob = dx[2,t] + normal(0, np.deg2rad(1.0))
            f_tr.write("EDGE_SE2 {} {} {} {} {} 0 0 0 0 0 0\n".format(t1, t2, edge_tr[0], edge_tr[1], dx[2,t]))
            f_ob.write("EDGE_SE2 {} {} {} {} {} 0 0 0 0 0 0\n".format(t1, t2, edge_ob[0], edge_ob[1], tan_ob))

            for j in range(NFEATURE):
                if norm(x[:2,t+1] - lm[:,j]) < RANGE:
                    if j not in lmdict.keys():
                        c += 1
                        lmdict[j] = t2 + c
                    edge_tr = rotation_matrix(-x[2,t+1]) @ (lm[:,j] - x[:2,t+1])
                    edge_ob = edge_tr + normal(0, 0.3, 2)
                    f_tr.write("EDGE_SE2_XY {} {} {} {} 0 0 0 0 0 0\n".format(t2, lmdict[j], edge_tr[0], edge_tr[1]))
                    f_ob.write("EDGE_SE2_XY {} {} {} {} 0 0 0 0 0 0\n".format(t2, lmdict[j], edge_ob[0], edge_ob[1]))

            t1 = t2
            t2 += 1 + c

    np.save(xnpy, x)
    np.save(lmnpy, lm)
    plt.plot(x[0,:], x[1,:])
    plt.scatter(lm[0,:], lm[1,:])
    plt.savefig(fig)
    plt.show()


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])


if __name__ == "__main__":
    main()
