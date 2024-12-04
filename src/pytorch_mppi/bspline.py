import numpy as np
import os


class BSpline:
    def __init__(self, n, d=3, num_T_pts=1024, name=""):
        self.num_T_pts = num_T_pts
        self.d = d
        self.n_pts = n
        self.m = self.d + self.n_pts
        self.u = np.pad(np.linspace(0., 1., self.m + 1 - 2 * self.d), self.d, 'edge')
        fname = f"{os.path.dirname(__file__)}/bspline_{name}_{self.n_pts}_{self.d}_{self.num_T_pts}.npy"
        if os.path.exists(fname):
            d = np.load(fname, allow_pickle=True).item()
            self.N = d["N"]
        else:
            self.N = self.calculate_N()
            np.save(fname, {"N": self.N}, allow_pickle=True)
        self.N = self.N.astype(np.float32)
        self.Ninv = np.linalg.pinv(self.N)

    def calculate_N(self):
        def N(n, t, i):
            if n == 0:
                if self.u[i] <= t < self.u[i + 1]:
                    return 1
                else:
                    return 0
            s = 0.
            if self.u[i + n] - self.u[i] != 0:
                s += (t - self.u[i]) / (self.u[i + n] - self.u[i]) * N(n - 1, t, i)
            if self.u[i + n + 1] - self.u[i + 1] != 0:
                s += (self.u[i + n + 1] - t) / (self.u[i + n + 1] - self.u[i + 1]) * N(n - 1, t, i + 1)
            return s

        T = np.linspace(0., 1., self.num_T_pts)
        Ns = [np.stack([N(self.d, t, i) for i in range(self.m - self.d)]) for t in T]
        Ns = np.stack(Ns, axis=0)
        Ns[-1, -1] = 1.
        return Ns[np.newaxis]