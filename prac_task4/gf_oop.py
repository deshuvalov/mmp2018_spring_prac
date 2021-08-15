import numpy as np
from functools import reduce

class F2qElement():
    def __init__(self, data, what, pm):
        if what == 'num':
            self.num = data
            self.pow = pm[data - 1, 0]
        elif what == 'pow':
            self.pow = data
            self.num = pm[data - 1, 1]
        else:
            raise NotImplementedError
        self.degree = len(bin(self.num)) - 3
        self.pm = pm
    
    def __add__(self, o):
        return F2qElement(self.num ^ o.num, 'num', self.pm)
        
    def __mul__(self, o):
        if self.num == 0 or o.num == 0:
            return F2qElement(0, num, self.pm)
        return F2qElement((self.pow + o.pow) % self.pm.shape[0], 'pow', self.pm)
        
    def __truediv__(self, o):
        if self.num == 0:
            return F2qElement(0, num, self.pm)
        if o.num == 0:
            raise ValueError
        return F2qElement((self.pow - o.pow) % self.pm.shape[0], 'pow', self.pm)

def gen_pow_matrix(primpoly):
    high_bit = 1 << (len(bin(primpoly)) - 3)
    pm = np.empty([high_bit - 1, 2], dtype=np.long)

    alpha_i = 0b10
    for i in range(high_bit - 1):
        pm[alpha_i - 1, 0] = i + 1
        pm[i, 1] = alpha_i

        alpha_i <<= 1
        if alpha_i & high_bit:
            alpha_i ^= primpoly

    return pm

class F2qPolynomial():
    def __init__(self, data, what, pm):
        if what == 'elem':
            self.coefs = data
        elif what == 'num':
            self.coefs = [F2qElement(num, 'num', pm) for num in data]
        elif what == 'pow':
            self.coefs = [F2qElement(pow, 'pow', pm) for pow in data]
        else:
            raise NotImplementedError
        self.degree = len(self.coefs) - 1
        self.pm = pm
        
    def trim_zeroes(self):
        for i, e in enumerate(self.coefs):
            if e.num > 0:
                break
        if i == len(self.coefs) or self.coefs[i].num == 0:
            self.coefs = [0]
        else:
            self.coefs = self.coefs[i:]
    
    def __add__(self, o):
        result = [self.coefs[i] + o.coefs[i] for i in range(min(len(self.coefs), len(o.coefs)))]
        if self.degree > o.degree:
            result = result + self.coefs[o.degree + 1:]
        elif o.degree > self.degree:
            result = result + o.coefs[self.degree + 1:]
        return F2qPolynomial(result, 'elem', self.pm)
        
    def __mul__(self, o):
        if self.coefs[0].num == 0 or o.coefs[0].num == 0:
            return F2qElement([0], num, pm)
        #return F2qElement((self.pow + o.pow) % pm.shape[0] - 1, 'pow', self.pm)
        
        self.trim_zeroes()
        o.trim_zeroes()

        result = [F2qElement(0, 'num', self.pm) for i in range(self.degree + o.degree + 1)]
        for i, value1 in enumerate(self.coefs):
            for j, value2 in enumerate(o.coefs):
                result[i + j] = result[i + j] + value1 * value2
        return F2qPolynomial(result, 'elem', self.pm)
 
       
    def __truediv__(self, o):
        if self.num == 0:
            return F2qElement(0, num, pm)
        if o.num == 0:
            raise ValueError
        return F2qElement((self.pow - o.pow) % pm.shape[0] - 1, 'pow', pm)

def gen_pow_matrix(primpoly):
    high_bit = 1 << (len(bin(primpoly)) - 3)
    pm = np.empty([high_bit - 1, 2], dtype=np.long)

    alpha_i = 0b10
    for i in range(high_bit - 1):
        pm[alpha_i - 1, 0] = i + 1
        pm[i, 1] = alpha_i

        alpha_i <<= 1
        if alpha_i & high_bit:
            alpha_i ^= primpoly

    return pm

def add(X, Y):
    return [[X[i][j] + Y[i][j] for j in range(len(X))] for i in range(len(X[0]))]


def sum(X, axis=0, keepdims=False):
    if axis == 0:
        return reduce((lambda x, y: x + y), X)
    else:
        return [sum(x, axis=0) for x in X]


def prod(X, Y):
    return [[X[i][j] * Y[i][j] for j in range(len(X))] for i in range(len(X[0]))]


def divide(X, Y, pm):
    return [[X[i][j] / Y[i][j] for j in range(len(X))] for i in range(len(X[0]))]


def linsolve(A, b, pm):
    A, b = A.copy(), b.copy()
    solution = []
    n = len(A)
    
    for i in range(n):
        if not A[i][i]:
            for nonzero_i in range(i, n):
                if A[nonzero_i][i] > 0:
                    break
            
            if nonzero_i == n or A[nonzero_i, i] == 0:
                return np.nan
                
            A[i], A[nonzero_i] = A[nonzero_i], A[i]
            b[i], b[nonzero_i] = b[nonzero_i], b[i]
        for j in range(i + 1, n):
            coef = A[j, i] / A[i, i]
            for k in range(i, n):
                A[j][k] = A[j][k] + A[i][k] * coef
            b[j] = b[j] + b[i] * coef
            
    solution[-1] = b[-1] / A[-1, -1]
    for i in reversed(range(n - 1)):
        solution[i] = (b[i] + sum(prod(A[i, i + 1:], solution[i + 1:], pm))) / A[i][i]
    return solution

def minpoly(x, pm):
    roots = set(x)
    for root in x:
        r_sqr = int(prod(root, root, pm))
        while r_sqr != root:
            roots.add(r_sqr)
            r_sqr = int(prod(r_sqr, r_sqr, pm))

    min_poly = np.asarray([1])
    for root in roots:
        min_poly = polyprod(min_poly, np.array([1, root]), pm)
    return min_poly, np.asarray(list(roots))
    

def polyval(p, x, pm):
    x_powers = np.ones([x.shape[0], p.shape[0]], dtype=np.long)
    for i in range(x_powers.shape[1] - 1):
        x_powers[:, i + 1] = prod(x_powers[:, i], x, pm)
    return sum(prod(p[::-1][np.newaxis, :], x_powers, pm), axis=1)


def polyadd(p1, p2):
    return p1 + p2


def polyprod(p1, p2, pm):
    return p1 * p2


def polydivmod(p1, p2, pm):
    p1.trim_zeroes()
    p2.trim_zeroes()
    
    if p2.coefs[0].num == 0:
        raise ValueError

    if p1.degree < p2.degree:
        return np.array([0]), p1

    result = [F2qElement(0, 'num', pm) for i in range(p1.degree - p2.degree + 1)]
    
    for i in range(len(result)):
        if not p1.coefs[i]:
            continue
        result[i] = p1.coefs[i] / p2.coefs[0]
        multiplier = F2qPolynomial(result[i:i + 1], 'elem') * p2
        multiplier.coefs = multiplier.coefs + [F2qElement(0, 'num', pm) for i in range(p1.degree - p2.degree - i)]
        p1 = p1 + multiplier
    p1.trim_zeroes()
    return result, p1


def euclid(p1, p2, pm, max_deg=0):
    coefficients = [
        [F2qElement(1, 'num', pm), F2qElement(0, 'num', pm)],
        [F2qElement(0, 'num', pm), F2qElement(1, 'num', pm)]
    ]
    while p2.degree > max_deg:
        q, r = polydivmod(p1, p2, pm)
        coefficients = [
            [coefficients[0][1], coefficients[0][0] + coefficients[0][1] * q],
            [coefficients[1][1], coefficients[1][0] + coefficients[1][1] * q]
        ]
        p1, p2 = p2, r
    return p2, coefficients[0][1], coefficients[1][1]
