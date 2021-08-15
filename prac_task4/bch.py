import numpy as np
import gf

class BCH:
    def __init__(self, n, t):
        self.n = n
        self.t = t
        self.q = int(np.log2(n + 1))

        with open('./primpoly.txt', 'r') as f:
            prim_polynoms = f.readline().strip().split(',')
        prim_polynoms = np.asarray([int(polynom) for polynom in prim_polynoms])
        prim_polynom = prim_polynoms[np.argmax(np.log2(prim_polynoms).astype(np.long) == self.q)]

        self.pm = gf.gen_pow_matrix(prim_polynom)
        self.R = self.pm[0:2*t, 1]
        self.g, _ = gf.minpoly(self.R, self.pm)

        self.m = self.g.shape[0] - 1
        self.k = self.n - self.m

    def _encode(self, u):
        assert self.k == u.shape[0], (self.k, u.shape[0])
        x_m = np.zeros(self.m + 1, dtype=np.long)
        x_m[0] = 1
        x_m_u = gf.polyprod(x_m, u, self.pm)
        _, mod = gf.polydivmod(x_m_u, self.g, self.pm)
        encoded = gf.polyadd(x_m_u, mod)
        encoded = np.concatenate([np.zeros([self.n - encoded.shape[0]], dtype=np.long), encoded])
        return encoded

    def encode(self, u):
        return np.array([self._encode(msg) for msg in u])

    def _decode(self, w, method):
        t = self.R.shape[0] // 2
        syndromes = gf.polyval(w, self.R, self.pm)
        if np.all(syndromes == 0):
            return w

        if method == 'pgz':
            lambda_ = np.nan
            for nu in reversed(range(1, t + 1)):
                a = np.array([[syndromes[j] for j in range(i, nu + i)] for i in range(nu)], dtype=np.long)
                b = np.array([syndromes[i] for i in range(nu, 2 * nu)], dtype=np.long)
                lambda_ = gf.linsolve(a, b, self.pm)
                if lambda_ is not np.nan:
                    break
            if lambda_ is np.nan:
                return np.full(self.n, np.nan, dtype=np.long)
            lambda_ = np.concatenate([lambda_, [1]])
        elif method == 'euclid':
            z = np.zeros([2 * (t + 1)], dtype=np.long)
            z[0] = 1
            syndromic_polynom = np.concatenate([syndromes[::-1], [1]])
            _, _, lambda_ = gf.euclid(z, syndromic_polynom, self.pm, max_deg=t)
        else:
            raise NotImplementedError

        n_roots = 0
        locators_values = gf.polyval(lambda_, np.arange(1, self.n + 1), self.pm)
        for i in range(self.n):
            if not locators_values[i]:
                position = self.n - self.pm[gf.divide(1, i + 1, self.pm) - 1, 0] - 1
                w[position] = 1 - w[position]
                n_roots += 1
        if n_roots != lambda_.shape[0] - 1:
            return np.full(self.n, np.nan, dtype=np.long)
        return w

    def decode(self, w, method='euclid'):
        return np.asarray([self._decode(x, method) for x in w])

    def dist(self):
        result = np.inf
        for value in range(1, 1 << self.k):
            block = np.asarray([int(digit) for digit in bin(value)[2:]])
            block = np.concatenate([np.zeros([self.k - block.shape[0]], dtype=np.long), block])
            result = min(np.count_nonzero(self._encode(block)), result)
        return int(result)
