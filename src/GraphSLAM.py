import numpy as np
from numpy.linalg import inv
from scipy import sparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

INF = 1e12


class GraphSLAM:

    def __init__(self, z, u, c, nfeature):
        """

        :param z: 3d list which represents measurement amount.
        :param u: 2d array which represents control amount.
        :param c: 2d list which represents correspondence.
        :param nfeature: # of features
        """

        self.z = z
        self.u = u
        self.c = c
        self.T = u.shape[1]
        self.nfeature = nfeature
        self.tau = [[] for _ in range(nfeature)]

        self.Rsigma1 = 0.1
        self.Rsigma2 = 0.1
        self.Rsigma3 = np.deg2rad(1.0)
        self.R = np.diag((self.Rsigma1, self.Rsigma2, self.Rsigma3))

        self.Qsigma1 = 0.1
        self.Qsigma2 = 0.1
        self.Q = np.diag((self.Qsigma1, self.Qsigma2))

        self.fig = plt.figure(figsize=(11,5))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

        self.max_iter = 1

    def main(self, method=0):
        print("GraphSLAM start!")
        print("# of landmarks   : {}".format(self.nfeature))
        print("# of time frames : {}".format(self.T))

        methods = ["numpy package as dense", "sum of block diagonal", "scipy package as sparse"]

        print("method for inverse calculation : {}".format(methods[method]))
        print("\n")

        n = 3 * (self.T + 1)

        mu, omega_til = self.predict(method)
        lms = mu[n:].reshape(len(mu[n:])//2, 2).T
        self.ax1.scatter(lms[0,:], lms[1,:], c="k", marker="x", label="landmark")

        self.ax2.pcolor(omega_til != 0, cmap=plt.cm.Blues)
        self.ax2.invert_yaxis()

        print("visualizing...")
        self.ax1.legend()
        self.ax1.set_title("predicted track by GraphSLAM")
        self.ax2.set_title("information matrix")
        # plt.show()
        print("done\n")

    def predict(self, method=0):
        n = 3 * (self.T + 1)
        mu_til = self._initialize(self.u)

        for t in range(self.T):
            for i in range(self.z[t].shape[1]):
                j = self.c[t][i]
                self.tau[j].append(t)

        print("initialize : {:.6f} [s]\n".format(self.time_initialize))
        self.ax1.plot(mu_til[0,:], mu_til[1,:], c="r", label="init")

        for i in range(self.max_iter):
            print("=====ITER {}=====\n".format(i + 1))

            omega, xi = self._linearize(self.u, self.z, self.c, mu_til)
            omega_til, xi_til = self._reduce(omega, xi, method)
            mu, sigma = self._solve(omega_til, xi_til, omega, xi, method)

            mu_til = mu[:n].reshape(self.T+1, 3).T
            self._time_display()
            self._visualize(mu[:n], i)

        return mu, omega_til

    def _initialize(self, u):
        """
        initialize the average pose vector mu.
        :param u: 2d array which represents control amount.
        :return: 2d array which represents average pose vector.
        """

        print("initializing...")
        t1 = time()

        mu = np.zeros((3, self.T+1), dtype=np.float)

        for t in range(self.T):
            mu[:,t+1] = self._func_g(u[:,t], mu[:,t])

        t2 = time()
        self.time_initialize = t2 - t1
        print("done\n")

        return mu

    def _linearize(self, u, z, c, mu):
        print("linearizing...")
        t1 = time()

        size = 3 * (self.T + 1) + 2 * self.nfeature
        omega = np.zeros((size, size), dtype=np.float)
        xi = np.zeros(size, dtype=np.float)

        omega[0:3,0:3] += np.array([[INF,0,0],
                                    [0,INF,0],
                                    [0,0,INF]])

        """
        update info matrix and vector at x_{t-1} and x_t
        """
        for t in range(self.T):
            x_hat = self._func_g(u[:,t], mu[:,t])
            G = self._jacobi_g(u[:,t], mu[:,t])

            A = np.hstack((-G, np.eye(3)))
            omega[3*t:3*t+6,3*t:3*t+6] += A.T @ inv(self.R) @ A

            B = x_hat - G @ mu[:,t]
            xi[3*t:3*t+6] += A.T @ inv(self.R) @ B

        """
        update info matrix and vector at x_t and m_j
        """
        for t in range(self.T):
            for i in range(z[t].shape[1]):
                j = c[t][i]
                mu_j = mu[:2,t+1] + self._rotation_matrix(mu[2,t+1]) @ z[t][:,i]

                z_hat = self._func_h(mu[:,t+1], mu_j)
                H = self._jacobi_h(mu[:,t+1], mu_j)

                pos_x = 3*t+3
                pos_m = 3*(self.T+1) + 2*j

                C = H.T @ inv(self.Q) @ H
                D = H.T @ inv(self.Q) @ (z[t][:,i] - z_hat + H @ np.hstack((mu[:,t+1], mu_j)))

                omega[pos_x:pos_x+3,pos_x:pos_x+3] += C[:3,:3]
                omega[pos_x:pos_x+3,pos_m:pos_m+2] += C[:3,3:5]
                omega[pos_m:pos_m+2,pos_x:pos_x+3] += C[3:5,:3]
                omega[pos_m:pos_m+2,pos_m:pos_m+2] += C[3:5,3:5]
                xi[pos_x:pos_x+3] += D[:3]
                xi[pos_m:pos_m+2] += D[3:5]

        t2 = time()
        self.time_linearize = t2 - t1
        print("done\n")

        return omega, xi

    def _reduce(self, omega, xi, method=0):
        print("reducing...")
        t1 = time()

        n = 3*(self.T + 1)
        omega_til = omega[:n, :n]
        xi_til = xi[:n]

        if method == 0:
            omega_til -= omega[:n, n:] @ inv(omega[n:, n:]) @ omega[n:, :n]
            xi_til -= omega[:n, n:] @ inv(omega[n:, n:]) @ xi[n:]
        elif method == 1:
            for j in range(self.nfeature):
                jrange = [n+2*j, n+2*j+1]
                beta = omega[:n, :][:, jrange]
                alpha = beta @ inv(omega[jrange, :][:, jrange])
                omega_til -= alpha @ beta.T
                xi_til -= alpha @ xi[jrange]
        elif method == 2:
            omega = sparse.csr_matrix(omega)
            omega_til -= omega[:n, n:] @ inv(omega[n:, n:]) @ omega[n:, :n]
            xi_til -= omega[:n, n:] @ inv(omega[n:, n:]) @ xi[n:]

        t2 = time()
        print("done\n")
        self.time_reduce = t2 - t1

        return omega_til, xi_til

    def _solve(self, omega_til, xi_til, omega, xi, method=0):
        print("solving...")
        t1 = time()

        n = 3 * (self.T + 1)

        mu = np.zeros_like(xi)
        sigma = inv(omega_til)
        mu[:n] = sigma @ xi_til

        if method == 0:
            sigma_m = inv(omega[n:, n:])
            mu[n:] = sigma_m @ (xi[n:] - omega[n:, :n] @ mu[:n])
        elif method == 1:
            for j in range(self.nfeature):
                self.tau[j] = np.array(self.tau[j])
                jrange = [n+2*j, n+2*j+1]
                sigma_j = inv(omega[jrange, :][:, jrange])
                tau = np.hstack((3*self.tau[j], 3*self.tau[j]+1, 3*self.tau[j]+2))
                mu[n+2*j:n+2*j+2] = sigma_j @ (xi[jrange] - omega[jrange, :][:, tau] @ mu[tau])
        elif method == 2:
            omega = sparse.csr_matrix(omega)
            sigma_m = inv(omega[n:, n:])
            mu[n:] = sigma_m @ (xi[n:] + omega[n:, :n] @ xi_til)

        t2 = time()
        self.time_solve = t2 - t1
        print("done\n")

        return mu, sigma

    def _visualize(self, mu, loop):
        """

        :param x: 2d array which represents predicted track
        :return:
        """

        colorlist = ["m", "b", "g"]
        x = [mu[i*3] for i in range(len(mu)//3)]
        y = [mu[i*3+1] for i in range(len(mu)//3)]
        self.ax1.plot(x, y, color=colorlist[loop%3], label="iter {}".format(loop + 1))

        return 0

    def _time_display(self):
        print("linearize : {:.6f} [s]".format(self.time_linearize))
        print("reduce    : {:.6f} [s]".format(self.time_reduce))
        print("solve     : {:.6f} [s]".format(self.time_solve))
        print("\n")

    # 運動学動作モデル g
    def _func_g(self, u, x_prev):
        x = np.zeros(3, dtype=np.float)
        x[:2] = x_prev[:2] + self._rotation_matrix(x_prev[2]) @ u[:2]
        x[2] = x_prev[2] + u[2]
        x[2] = x[2] % (2 * np.pi)

        return x

    # 関数gのxに対するヤコビ行列
    def _jacobi_g(self, u, x):

        G = np.eye(3, dtype=np.float)
        P = np.array([[-np.sin(x[2]), -np.cos(x[2])],
                      [np.cos(x[2]),  -np.sin(x[2])]])
        G[:2,2] = P @ u[:2]

        return G

    # 計測関数 h
    def _func_h(self, x, m):
        Rot = self._rotation_matrix(-x[2])

        z = Rot @ (m[:2] - x[:2])

        return z

    # 関数hの[x_t, m_j]におけるヤコビ行列
    def _jacobi_h(self, x, m):
        h = np.zeros((2,5))
        Rot = self._rotation_matrix(-x[2])
        P = np.array([[-np.sin(x[2]),  np.cos(x[2])],
                      [-np.cos(x[2]), -np.sin(x[2])]])
        h[:2,:2] = -1 * Rot
        h[:2,3:5] = Rot
        h[:2,2] = P @ (m[:2] - x[:2])

        return h

    def _rotation_matrix(self, theta):
        Rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

        return Rot
