import numpy as np
from numpy.linalg import inv

INF = 1e9


class GraphSLAM:

    def __init__(self, z, u, f):
        """

        :param z: 3 dimentional array which represents measurement amount.
        :param u: 2 dimentional array which represents control amount.
        :param f: sampling rate
        """

        self.z = z
        self.u = u
        self.T = len(u)
        self.f = f
        self.nfeature = 0
        self.tau = dict()

        self.Rsigma1 = 0.1
        self.Rsigma2 = 0.1
        self.Rsigma3 = np.deg2rad(1.0)
        self.R = np.diag((self.Rsigma1, self.Rsigma2, self.Rsigma3))

        self.Qsigma1 = 0.1
        self.Qsigma2 = 0.1
        self.Qsigma3 = 0.1
        self.Q = np.diag((self.Qsigma1, self.Qsigma2, self.Qsigma3))

        self.max_iter = 3

    def predict(self, u, z):
        c = np.random.randint(3)
        mu = self._initialize(u)
        omega, xi = self._linearize(u, z, c, mu)
        omega_til, xi_til = self._reduce(omega, xi)
        mu, sigma = self._solve(omega_til, xi_til, omega, xi)
        for _ in range(self.max_iter):
            n = 3 * (self.T + 1)
            omega, xi = self._linearize(u, z, c, mu[:n])
            omega_til, xi_til = self._reduce(omega, xi)
            mu, sigma = self._solve(omega_til, xi_til, omega, xi)

        return mu

    def _initialize(self, u):
        """
        initialize the average pose vector mu.
        :param u: 2 dimentional array which represents control amount.
        :return: 2 dimentional array which represents average pose vector.
        """
        mu = np.zeros((3, self.T+1), dtype=np.float)
        d_mu = np.zeros(3)

        for t in range(self.T):
            d_mu[0] = (- np.sin(mu[2,t-1]) + np.sin(mu[2,t-1]+u[1]/self.f)) * (u[0]/u[1])
            d_mu[1] = (np.cos(mu[2,t-1]) - np.cos(mu[2,t-1]+u[1]/self.f)) * (u[0]/u[1])
            d_mu[2] = u[1] / self.f

            mu[:,t+1] = mu[:,t] + d_mu

        return mu

    def _linearize(self, u, z, c, mu):
        omega = np.zeros((3*(self.T+1),3*(self.T+1)),dtype=np.float)
        xi = np.zeros((3*(self.T+1),3*(self.T+1)), dtype=np.float)

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
            for i in range(len(z[:,t])):
                j = c[i,t]
                if j in self.tau.keys():
                    self.tau[j].append(t)
                else:
                    self.tau[j] = [t]
                z_hat = self._func_h(mu[:,t+1], mu[:,j])
                H = self._jacobi_h(mu[:,t+1], mu[:,j])

                pos_x = 3*t
                pos_m = 3*(self.T+1) + 2*j

                C = H.T @ inv(self.Q) @ H
                D = H.T @ inv(self.Q) @ (z[:,t]-z_hat+H@np.hstack((mu[:,t+1], mu[:,j])))
                if not j < self.nfeature:
                    size = 3*(self.T+1) + 2*(j+1)
                    tmp_mat = np.zeros((size,size))
                    tmp_mat[:omega.shape[0], :omega.shape[1]] = omega
                    omega = tmp_mat
                    tmp_vec = np.zeros(size)
                    tmp_vec[:xi.shape[0]] = xi
                    xi = tmp_vec
                    self.nfeature = j+1
                omega[pos_x:pos_x+3,pos_x:pos_x+3] += C[:3,:3]
                omega[pos_x:pos_x+3,pos_m:pos_m+2] += C[:3,3:5]
                omega[pos_m:pos_m+2,pos_x:pos_x+3] += C[3:5,:3]
                omega[pos_m:pos_m+2,pos_m:pos_m+2] += C[3:5,3:5]
                xi[pos_x:pos_x+3] += D[:3]
                xi[pos_m:pos_m+2] += D[3:5]

        return omega, xi

    def _reduce(self, omega, xi):
        n = 3*(self.T + 1)
        omega_til = omega[:n, :n]
        xi_til = xi[:n]

        for j in range(self.nfeature):
            omega_til -= omega[:n, n+2*j:n+2*j+2] @ omega[n+2*j:n+2*j+2, n+2*j:n+2*j+2] @ omega[n+2*j:n+2*j+2, :n]
            xi_til -= omega[:n, n+2*j:n+2*j+2] @ omega[n+2*j:n+2*j+2, n+2*j:n+2*j+2] @ xi[:n]
        return omega_til, xi_til

    def _solve(self, omega_til, xi_til, omega, xi):
        n = 3 *(self.T + 1)

        mu = np.zeros_like(xi)
        sigma = inv(omega_til)
        mu[:n] = sigma @ xi_til

        for j in range(self.nfeature):
            sigma_j = inv(omega[n+2*j:n+2*j+2, n+2*j:n+2*j+2])
            rang = np.hstack((self.tau[j], self.tau[j]+1, self.tau[j]+2))
            mu[2*j:2*j+2] = sigma_j @ (xi[2*j:2*j+2] + sigma[2*j:2*j+2,:][:,rang] @ mu[rang])
        return mu, sigma

    def _correspondence_test(self, omega, mu, sigma, j, k):
        tau_jk = np.hstack((self.tau[j], self.tau[j]+1, self.tau[j]+2, self.tau[k], self.tau[k]+1, self.tau[k]+2))
        jk = np.array([j, j+1, k, k+1]) + 3*(self.T + 1)

        omega_jk = omega[jk,:][:,jk] - omega[jk,:][:,tau_jk] @ sigma[tau_jk,:][:,tau_jk] @ omega[tau_jk,:][:,jk]
        xi_jk = omega_jk @ mu[jk]

        return 0

    # 運動学動作モデル g
    def _func_g(self, u, x_prev):
        d_x = np.zeros(3)
        d_x[0] = (- np.sin(x_prev[2]) + np.sin(x_prev[2] + u[1] / self.f)) * (u[0] / u[1])
        d_x[1] = (np.cos(x_prev[2]) - np.cos(x_prev[2] + u[1] / self.f)) * (u[0] / u[1])
        d_x[2] = u[1] / self.f

        x = x_prev + d_x

        return x

    # 関数gのxに対するヤコビ行列
    def _jacobi_g(self, u, x):
        G = np.eye(3)

        G[0,2] = (- np.cos(x[2]) + np.cos(x[2] + u[1] / self.f)) * (u[0] / u[1])
        G[1,2] = (- np.sin(x[2]) + np.sin(x[2] + u[1] / self.f)) * (u[0] / u[1])

        return G

    # 計測関数 h
    def _func_h(self, x, m, predict=False):
        z = np.zeros(3, dtype=np.float)
        Rot = self._rotation_matrix(-x[2])

        z[:2] = Rot @ (m[:2] - x[:2])
        z[2] = 0 if predict else m[2]

        return z

    # 関数hの[x_t, m_j]におけるヤコビ行列
    def _jacobi_h(self, x, m):
        h = np.zeros((3,6))
        Rot = self._rotation_matrix(x[2])
        P = np.array([[-np.sin(x[2]), -np.cos(x[2])],
                      [np.cos(x[2]), -np.sin(x[2])]])
        h[:2,:2] = -1 * Rot
        h[:2,3:5] = Rot
        h[:2,2] = P @ (m[:2] - x[:2])
        h[2,5] = 1

        return h

    def _rotation_matrix(self, theta):
        Rot = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        return Rot

