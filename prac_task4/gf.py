import numpy as np


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
    return X ^ Y


def sum(X, axis=0, keepdims=False):
    return np.bitwise_xor.reduce(X, axis=axis, keepdims=keepdims)


def prod(X, Y, pm):
    X_pow, Y_pow = pm[X - 1, 0], pm[Y - 1, 0]
    product = pm[np.mod(X_pow + Y_pow, pm.shape[0]) - 1, 1]
    if isinstance(product, np.integer) or product.size == 1:
        product = np.asarray(product)
        
    product[(X == 0) | (Y == 0)] = 0
    return product


def divide(X, Y, pm):
    assert np.all(Y != 0)
    X_pow, Y_pow = pm[X - 1, 0], pm[Y - 1, 0]
    quotient = pm[np.mod(X_pow - Y_pow, pm.shape[0]) - 1, 1]
    if isinstance(quotient, np.integer) or quotient.size == 1:
        quotient = np.asarray(quotient)
        
    quotient[(X == 0)] = 0
    return quotient


def linsolve(A, b, pm):
    A, b = np.copy(A), np.copy(b)
    solution = np.empty_like(b)
    n = A.shape[0]
    
    for i in range(n):
        if not A[i,i]:
            nonzero_i = np.argmax(A[i:, i] > 0)
            if A[i + nonzero_i, i] == 0:
                return np.nan
            nonzero_i += i
            A[:, i:][[i, nonzero_i]] = A[:, i:][[nonzero_i, i]]
            b[i], b[nonzero_i] = b[nonzero_i], b[i]
        for j in range(i + 1, n):
            coef = divide(A[j, i], A[i, i], pm)
            A[j, i:] = add(A[j, i:], prod(A[i, i:], coef, pm))
            b[j] = add(b[j], prod(b[i], coef, pm))
            
    solution[-1] = divide(b[-1], A[-1, -1], pm)
    for i in reversed(range(n - 1)):
        solution[i] = divide(add(b[i], sum(prod(A[i, i + 1:], solution[i + 1:], pm))), A[i, i], pm)
    return solution


def trim_zeroes(p):
    if np.sum(p > 0) == 0:
        return np.asarray([0])
    return p[np.argmax(p > 0):]


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


def polyadd(p1, p2, trim=True):
    if p1.shape[0] > p2.shape[0]:
        p2 = np.concatenate([np.zeros(p1.shape[0] - p2.shape[0], dtype=np.long), p2])
    else:
        p1 = np.concatenate([np.zeros(p2.shape[0] - p1.shape[0], dtype=np.long), p1])
    result = add(p1, p2)
    if trim:
        return trim_zeroes(result)
    return result


def polyprod(p1, p2, pm):
    p1 = trim_zeroes(p1)
    p2 = trim_zeroes(p2)

    result = np.zeros([p1.shape[0] + p2.shape[0] - 1], dtype=np.long)
    for i, value1 in enumerate(p1):
        for j, value2 in enumerate(p2):
            result[i + j] = add(result[i + j], prod(value1, value2, pm))
    return trim_zeroes(result)


def polydivmod(p1, p2, pm):
    p1 = trim_zeroes(p1)
    p2 = trim_zeroes(p2)
    
    if p2[0] == 0:
        raise ValueError

    if p1.shape[0] < p2.shape[0]:
        return np.array([0]), p1

    result = np.zeros(p1.shape[0] - p2.shape[0] + 1, dtype=np.long)
    for i in range(result.shape[0]):
        if not p1[i]:
            continue
        result[i] = divide(p1[i], p2[0], pm)
        multiplier = np.concatenate([
            polyprod(result[i:i + 1], p2, pm),
            np.zeros([p1.shape[0] - p2.shape[0] - i], dtype=np.long)
        ])
        p1 = polyadd(p1, multiplier, trim=False)
    return result, trim_zeroes(p1)


def euclid(p1, p2, pm, max_deg=0):
    coefficients = [
        [np.array([1]), np.array([0])],
        [np.array([0]), np.array([1])]
    ]
    while p2.shape[0] - 1 > max_deg:
        q, r = polydivmod(p1, p2, pm)
        coefficients = [
            [coefficients[0][1], polyadd(coefficients[0][0], polyprod(coefficients[0][1], q, pm))],
            [coefficients[1][1], polyadd(coefficients[1][0], polyprod(coefficients[1][1], q, pm))]
        ]
        p1, p2 = p2, r
    return p2, coefficients[0][1], coefficients[1][1]
