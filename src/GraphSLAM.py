import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from tqdm import tqdm

INF = 1e9


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

        self.max_iter = 3

    def predict(self, method=0):
        n = 3 * (self.T + 1)
        mu = self._initialize(self.u)
        for _ in range(self.max_iter):
            omega, xi = self._linearize(self.u, self.z, self.c, mu[:n])
            omega_til, xi_til = self._reduce(omega, xi, method)
            mu, sigma = self._solve(omega_til, xi_til, omega, xi, method)

        return mu

    def _initialize(self, u):
        """
        initialize the average pose vector mu.
        :param u: 2d array which represents control amount.
        :return: 2d array which represents average pose vector.
        """
        mu = np.zeros((3, self.T+1), dtype=np.float)

        for t in range(self.T):
            mu[:,t+1] = self._func_g(u[:,t], mu[:,t])

        return mu

    def _linearize(self, u, z, c, mu):
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
            x_hat = self._func_g(mu[:,t], u[:,t])
            G = self._jacobi_g(mu[:,t], u[:,t])

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
                self.tau[j].append(t)

                z_hat = self._func_h(mu[:,t+1], mu[:,j])
                H = self._jacobi_h(mu[:,t+1], mu[:,j])

                pos_x = 3*t
                pos_m = 3*(self.T+1) + 2*j

                C = H.T @ inv(self.Q) @ H
                D = H.T @ inv(self.Q) @ (z[t][:,i]-z_hat+H@np.hstack((mu[:,t+1], mu[:2,j])))
                """
                if not j < self.nfeature:
                    size = 3*(self.T+1) + 2*(j+1)
                    tmp_mat = np.zeros((size,size))
                    tmp_mat[:omega.shape[0], :omega.shape[1]] = omega
                    omega = tmp_mat
                    tmp_vec = np.zeros(size)
                    tmp_vec[:xi.shape[0]] = xi
                    xi = tmp_vec
                    self.nfeature = j+1
                """
                omega[pos_x:pos_x+3,pos_x:pos_x+3] += C[:3,:3]
                omega[pos_x:pos_x+3,pos_m:pos_m+2] += C[:3,3:5]
                omega[pos_m:pos_m+2,pos_x:pos_x+3] += C[3:5,:3]
                omega[pos_m:pos_m+2,pos_m:pos_m+2] += C[3:5,3:5]
                xi[pos_x:pos_x+3] += D[:3]
                xi[pos_m:pos_m+2] += D[3:5]

        return omega, xi

    def _reduce(self, omega, xi, method=0):
        n = 3*(self.T + 1)
        omega_til = omega[:n, :n]
        xi_til = xi[:n]

        if method == 0:
            omega_til -= omega[:n, n:] @ inv(omega[n:, n:]) @ omega[n:, :n]
            xi_til -= omega[:n, n:] @ inv(omega[n:, n:]) @ xi[n:]
        elif method == 1:
            for j in tqdm(range(self.nfeature)):
                jrange = [n+2*j, n+2*j+1]
                beta = omega[:n, :][:, jrange]
                alpha = beta @ inv(omega[jrange, :][:, jrange])
                omega_til -= np.dot(alpha, beta.T)
                xi_til -= alpha @ xi[jrange]
        return omega_til, xi_til

    def _solve(self, omega_til, xi_til, omega, xi, method=0):
        n = 3 * (self.T + 1)

        mu = np.zeros_like(xi)
        sigma = inv(omega_til)
        mu[:n] = sigma @ xi_til

        if method == 0:
            sigma_m = inv(omega[n:, n:])
            mu[n:] = sigma_m @ (xi[n:] + omega[n:, :n] @ xi_til)
        elif method == 1:
            for j in range(self.nfeature):
                self.tau[j] = np.array(self.tau[j])
                jrange = [n+2*j, n+2*j+1]
                sigma_j = inv(omega[jrange, :][:, jrange])
                tau = np.hstack((3*self.tau[j], 3*self.tau[j]+1, 3*self.tau[j]+2))
                mu[n+2*j:n+2*j+2] = sigma_j @ (xi[jrange] + omega[jrange, :][:, tau] @ mu[tau])
        return mu, sigma

    def _visualize(self):
        return 0

    # 運動学動作モデル g
    def _func_g(self, u, x_prev):
        x = np.zeros(3, dtype=np.float)
        """
        d_x = np.zeros(3)
        d_x[0] = (- np.sin(x_prev[2]) + np.sin(x_prev[2] + u[1] / self.f)) * (u[0] / u[1])
        d_x[1] = (np.cos(x_prev[2]) - np.cos(x_prev[2] + u[1] / self.f)) * (u[0] / u[1])
        d_x[2] = u[1] / self.f
        """
        x[:2] = x_prev[:2] + self._rotation_matrix(x_prev[2]) @ u[:2]
        x[2] = x_prev[2] + u[2]

        return x

    # 関数gのxに対するヤコビ行列
    def _jacobi_g(self, u, x):
        """
        G = np.eye(3)

        G[0,2] = (- np.cos(x[2]) + np.cos(x[2] + u[1] / self.f)) * (u[0] / u[1])
        G[1,2] = (- np.sin(x[2]) + np.sin(x[2] + u[1] / self.f)) * (u[0] / u[1])
        """

        G = np.eye(3, dtype=np.float)
        P = np.array([[-np.sin(x[2]), -np.cos(x[2])],
                      [np.cos(x[2]), -np.sin(x[2])]])
        G[:2,2] = P @ u[:2]

        return G

    # 計測関数 h
    def _func_h(self, x, m):
        z = np.zeros(2, dtype=np.float)
        Rot = self._rotation_matrix(-x[2])

        z[:2] = Rot @ (m[:2] - x[:2])

        return z

    # 関数hの[x_t, m_j]におけるヤコビ行列
    def _jacobi_h(self, x, m):
        h = np.zeros((2,5))
        Rot = self._rotation_matrix(x[2])
        P = np.array([[-np.sin(x[2]), -np.cos(x[2])],
                      [np.cos(x[2]), -np.sin(x[2])]])
        h[:2,:2] = -1 * Rot
        h[:2,3:5] = Rot
        h[:2,2] = P @ (m[:2] - x[:2])

        return h

    def _rotation_matrix(self, theta):
        Rot = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        return Rot

    def test(self):
        n = 3 * (self.T + 1)
        mu = self._initialize(self.u)
        plt.scatter(mu[0,:], mu[1,:])
        plt.show()
        # print(mu)
        # omega, xi = self._linearize(self.u, self.z, self.c, mu[:n])
        # print(omega)
        # print(xi)
        # omega_til, xi_til = self._reduce(omega, xi, method=0)
        # print(omega_til)
        # print(xi_til)
        # mu, sigma = self._solve(omega_til, xi_til, omega, xi, method=0)
        # print(mu)
        # print(sigma)
        # plt.scatter(mu[[3*i for i in range(self.T+1)]], mu[[3*i+1 for i in range(self.T+1)]])
        # plt.show()
