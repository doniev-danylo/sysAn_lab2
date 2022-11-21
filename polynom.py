import copy

import numpy as np


class Polynom:
    def __init__(self, deg, coeff):
        self.deg = deg
        self.coeff = coeff

    def sum(self, rhs):
        result = Polynom(max(rhs.deg, self.deg), np.zeros(max(rhs.deg, self.deg) + 1))
        coeff1 = rhs.coeff[::-1]
        coeff2 = self.coeff[::-1]
        for i in range(max(rhs.deg, self.deg) + 1):
            cand1 = 0
            if i <= rhs.deg:
                cand1 = coeff1[i]

            cand2 = 0
            if i <= self.deg:
                cand2 = coeff2[i]

            result.coeff[i] = cand1 + cand2

        result.coeff = result.coeff[::-1]
        return result

    def mul(self, rhs):
        result = Polynom(rhs.deg + self.deg, np.zeros(rhs.deg + self.deg + 1))
        coeff1 = rhs.coeff[::-1]
        coeff2 = self.coeff[::-1]
        result_coeff = np.zeros(rhs.deg + self.deg + 1)
        for deg1, val1 in enumerate(coeff1):
            for deg2, val2 in enumerate(coeff2):
                result_coeff[deg1 + deg2] += val1 * val2

        result.coeff = result_coeff[::-1]
        return result

    def mul_for_const(self, k):
        result = Polynom(self.deg, copy.deepcopy(self.coeff))
        for i in range(self.deg + 1):
            result.coeff[i] = result.coeff[i] * k
        return result

    def calc(self, point):
        result = 0
        for deg, coeff in enumerate(self.coeff[::-1]):
            result += coeff * (point ** deg)
        return result

    def factorial(n):
        if n == 0 or n == 1:
            return 1

        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    def cnk(n, k):
        return Polynom.factorial(n) / (Polynom.factorial(n - k) * Polynom.factorial(k))

    def calc_binom(a, b, n):
        result = Polynom(0, [0])
        for k in range(n + 1):
            c = Polynom.cnk(n, k) * (a ** k) * (b ** (n - k))
            coeff_arr = np.zeros(k + 1)
            coeff_arr[0] = c
            result = result.sum(Polynom(k, coeff_arr))
        return result

    def substitution(self, a, b):
        result = Polynom(0, [0])
        for deg, coeff in enumerate(self.coeff[::-1]):
            result = result.sum(Polynom.calc_binom(a, b, deg).mul_for_const(coeff))
        return result

    def add_const(self, k):
        result = Polynom(self.deg, copy.deepcopy(self.coeff))
        result.coeff[-1] += k
        return result

    def print(self, name='x'):
        answer = ""
        for idx, coeff in enumerate(self.coeff):
            if abs(coeff) > 1e-6:
                if idx != self.deg:
                    answer += "%.3e * %s ^ %d + " % (coeff, name, self.deg - idx)
                else:
                    answer += "%.3e" % (coeff)
        return answer
