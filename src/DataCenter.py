import os
import pandas as pd
import numpy as np


def read(fname):
    """

    :param fname: the name of file
    :type fname: str
    :return: z, c, u, nfeature
    """
    data = pd.read_table(fname, sep=" ", header=None)
    # data = data[:500]
    measure = data[data[0]=="EDGE_SE2_XY"]
    control = data[data[0]=="EDGE_SE2"]

    # create control array u
    u = np.array(control.iloc[:,3:6]).T

    # create measurement array z
    time = 0
    tmp = 0
    nfeature = 0
    fdict = dict()
    c = [[] for _ in range(u.shape[1])]
    z = [np.array([[], []]) for _ in range(u.shape[1])]
    for _, row in measure.iterrows():
        if tmp != row[1]:
            tmp = row[1]
            time = tmp - nfeature - 1
        if tmp <= row[2]:
            fdict[row[2]] = nfeature
            nfeature += 1
        #print(row[2])
        c[time].append(fdict[row[2]])
        z[time] = np.hstack((z[time], np.array([[row[3]],[row[4]]])))

    return z, u, c, nfeature


if __name__ == '__main__':
    fname = "../data/sample0/input.txt"
    z, c, u, n = read(fname)
    print(n)
    print(c[:10])
    for i in range(10):
        print(z[i])
