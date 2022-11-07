import copy
import os
from tkinter import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_C(X, Y, degrees_of_polynomials, size_of_vectors, calc_polynom_in_point, lambdas, alphas):
    C = []
    phi_count = Y.shape[1]
    N = Y.shape[0]
    for phi_idx in range(phi_count):
        current_C = []
        for sample_number in range(N):
            coeff = []
            for i, X_i in enumerate(X):
                current_coeff = 0
                idx = 0
                for z in range(i):
                    idx += size_of_vectors[z] * (degrees_of_polynomials[z] + 1)
                for j in range(size_of_vectors[i]):
                    for degree in range(degrees_of_polynomials[i] + 1):
                        current_coeff += alphas[phi_idx][i][j] * lambdas[phi_idx][idx] * calc_polynom_in_point(
                            X_i[sample_number][j], degree)
                        idx += 1
                coeff.append(current_coeff)
            current_C.append(coeff)
        C.append(current_C)

    result = [np.linalg.lstsq(C[k], Y[:, k], rcond=None)[0] for k in range(phi_count)]
    return result


def norm_all_matrix(a):
    min_element = a.min()
    max_element = a.max()
    result = copy.deepcopy(a)
    result = result.astype(np.float32)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = (result[i, j] - min_element) / (max_element - min_element)
    return result


def find_lambdas(X, Y, degrees_of_polynomials, size_of_vectors, calc_polynom_in_point):
    A = []
    N = Y.shape[0]
    for sample_number in range(N):
        current_row = []
        for i, X_i in enumerate(X):
            current_row = np.hstack([current_row, [calc_polynom_in_point(X_i[sample_number][j], degree)
                                                   for j in range(size_of_vectors[i])
                                                   for degree in range(degrees_of_polynomials[i] + 1)]])
        A.append(current_row)

    lambdas = [np.linalg.lstsq(A, Y[:, k], rcond=None)[0] for k in range(Y.shape[1])]
    return lambdas


def star_chebyshev_polynom(degree):
    if degree == 0:
        return Polynom(0, [0.5])
    values = [0] * (degree + 1)
    values[0] = Polynom(0, [1])
    values[1] = Polynom(1, [2, -1])
    for current_degree in range(2, degree + 1):
        values[current_degree] = values[current_degree - 1].mul(Polynom(1, [4, -2])).sum(
            values[current_degree - 2].mul_for_const(-1))
    return values[degree]


def star_chebyshev_polynom_in_point(x, degree):
    if degree == 0:
        return 0.5
    else:
        return chebyshev_polynom_in_point(2 * x - 1, degree)


def chebyshev_polynom_in_point(x, degree):
    values = np.zeros(degree + 1)
    values[0] = 1
    values[1] = x
    for current_degree in range(2, degree + 1):
        values[current_degree] = 2 * x * values[current_degree - 1] - values[current_degree - 2]
    return values[degree]


def find_alphas(X, Y, degrees_of_polynomials, size_of_vectors, calc_polynom_in_point, lambdas):
    alphas = []
    phi_count = Y.shape[1]
    N = Y.shape[0]
    for phi_idx in range(phi_count):
        B = []
        for i, X_i in enumerate(X):
            current_B = []
            for sample_number in range(N):
                current_row = []
                idx = 0
                for z in range(i):
                    idx += size_of_vectors[z] * (degrees_of_polynomials[z] + 1)
                for j in range(size_of_vectors[i]):
                    c = 0
                    for degree in range(degrees_of_polynomials[i] + 1):
                        c += calc_polynom_in_point(X_i[sample_number][j], degree) * lambdas[phi_idx][idx]
                        idx += 1
                    current_row.append(c)
                current_B.append(current_row)
            B.append(current_B)
        alphas.append([np.linalg.lstsq(A, Y[:, phi_idx], rcond=None)[0] for A in B])
    return alphas


def lagger_polynom_in_point(x, degree):
    if degree == 0:
        return 0.5
    else:
        values = np.zeros(degree + 1)
        values[0] = 1
        values[1] = -x + 1
        for curr_degree in range(2, degree + 1):
            n = curr_degree - 1
            values[curr_degree] = (2 * n + 1 - x) * values[curr_degree - 1] - n * n * values[curr_degree - 2]
        return values[degree]


def lagger_polynom(degree):
    if degree == 0:
        return Polynom(0, [0.5])
    else:
        values = [0] * (degree + 1)
        values[0] = Polynom(0, [1])
        values[1] = Polynom(1, [-1, 1])
        for curr_degree in range(2, degree + 1):
            n = curr_degree - 1
            values[curr_degree] = values[curr_degree - 1].mul_for_const(2 * n + 1).sum(
                values[curr_degree - 1].mul(Polynom(1, [-1, 0]))).sum(
                values[curr_degree - 2].mul_for_const(-1 * n * n))
        return values[degree]


def hermit_polynom_in_point(x, degree):
    if degree == 0:
        return 1
    else:
        values = np.zeros(degree + 1)
        values[0] = 1
        values[1] = x
        for curr_degree in range(2, degree + 1):
            n = curr_degree - 1
            values[curr_degree] = x * values[curr_degree - 1] - n * values[curr_degree - 2]
        return values[degree]


def hermit_polynom(degree):
    if degree == 0:
        return Polynom(0, [1])
    else:
        values = [0] * (degree + 1)
        values[0] = Polynom(0, [1])
        values[1] = Polynom(1, [1, 0])
        for curr_degree in range(2, degree + 1):
            n = curr_degree - 1
            values[curr_degree] = values[curr_degree - 1].mul(Polynom(1, [1, 0])).sum(
                values[curr_degree - 2].mul_for_const(-1 * n))
        return values[degree]


def legender_polynom_in_point(x, degree):
    if degree == 0:
        return 0.5
    else:
        values = np.zeros(degree + 1)
        values[0] = 1
        values[1] = x
        for curr_degree in range(2, degree + 1):
            n = curr_degree - 1
            values[curr_degree] = ((2 * n + 1) * x * values[curr_degree - 1] - n * values[curr_degree - 2]) * (
                    1 / (n + 1))
        return values[degree]


def legender_polynom(degree):
    if degree == 0:
        return Polynom(0, [0.5])
    else:
        values = [0] * (degree + 1)
        values[0] = Polynom(0, [1])
        values[1] = Polynom(1, [1, 0])
        for curr_degree in range(2, degree + 1):
            n = curr_degree - 1
            values[curr_degree] = values[curr_degree - 1].mul(Polynom(1, [1, 0])).mul_for_const(2 * n + 1).sum(
                values[curr_degree - 2].mul_for_const(-1 * n))
            values[curr_degree] = values[curr_degree].mul_for_const(1 / (n + 1))
        return values[degree]


def star_legender_polynom_in_point(point, degree):
    return legender_polynom_in_point(2 * point - 1, degree)


def star_legender_polynom(degree):
    if degree == 0:
        return Polynom(0, [0.5])
    else:
        return legender_polynom(degree).substitution(2, -1)


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


class Interface:
    def __init__(self):
        self.root = Tk()
        self.root.resizable(False, False)
        self.root.title("Вайнер_Чернюк_2_Лаба")
        self.root["bg"] = "grey"
        self.root.iconbitmap('calc.ico')

        Label(text="Дані", fg="blue", bg="gray", font=('Haettenschweiler 15 underline')).grid(row=0, column=0,
                                                                                              columnspan=2)

        Label(text="Розмір вибірки: ", bg="gray", font=("Comic Sans MS", 9)).grid(row=1, column=0, sticky="e")
        self.dataSize = IntVar()
        self.dataSize.set(40)
        Entry(textvariable=self.dataSize, width=4, justify='center').grid(row=1, column=1, sticky="w")

        Label(text="Файл вихідних даних X: ", bg="gray", font=("Comic Sans MS", 9)).grid(row=2, column=0, sticky="e")
        self.dataInputX = StringVar()
        self.dataInputX.set("X_data.csv")
        Entry(textvariable=self.dataInputX, width=9, justify='center').grid(row=2, column=1, sticky="w")

        Label(text="Файл вихідних даних Y: ", bg="gray", font=("Comic Sans MS", 9)).grid(row=3, column=0, sticky="e")
        self.dataInputY = StringVar()
        self.dataInputY.set("Y_data.csv")
        Entry(textvariable=self.dataInputY, width=9, justify='center').grid(row=3, column=1, sticky="w")

        Label(text="Файл результатів: ", bg="gray", font=("Comic Sans MS", 9)).grid(row=6, column=0, sticky="w")
        self.dataOutput = StringVar()
        self.dataOutput.set("output.txt")
        self.dataOutput.get()
        Entry(textvariable=self.dataOutput, width=20, justify='center').grid(row=6, column=0, sticky="e", columnspan=2)

        Label(text="Вектори", fg="blue", bg="gray", font=('Haettenschweiler 15 underline')).grid(row=0, column=2,
                                                                                                 columnspan=2)

        Label(text="Розмірність X₁: ", bg="gray", font=("Comic Sans MS", 9)).grid(row=1, column=2, sticky="e")
        self.sizeX1 = IntVar()
        self.sizeX1.set(2)
        Entry(textvariable=self.sizeX1, width=4, justify='center').grid(row=1, column=3, sticky="w")

        Label(text="Розмірність X₂: ", bg="gray", font=("Comic Sans MS", 9)).grid(row=2, column=2, sticky="e")
        self.sizeX2 = IntVar()
        self.sizeX2.set(2)
        Entry(textvariable=self.sizeX2, width=4, justify='center').grid(row=2, column=3, sticky="w")

        Label(text="Розмірність X₃: ", bg="gray", font=("Comic Sans MS", 9)).grid(row=3, column=2, sticky="e")
        self.sizeX3 = IntVar()
        self.sizeX3.set(2)
        Entry(textvariable=self.sizeX3, width=4, justify='center').grid(row=3, column=3, sticky="w")

        Label(text="Розмірність Y: ", bg="gray", font=("Comic Sans MS", 9)).grid(row=4, column=2, sticky="e")
        self.sizeY = IntVar()
        self.sizeY.set(5)
        Entry(textvariable=self.sizeY, width=4, justify='center').grid(row=4, column=3, sticky="w")

        Label(text="Степені полінома", fg="blue", bg="gray", font=('Haettenschweiler 15 underline')).grid(row=0,
                                                                                                          column=4,
                                                                                                          columnspan=2)

        Label(text="Для X₁: ", bg="gray", font=("Comic Sans MS", 9)).grid(row=1, column=4, sticky="e")
        self.st1 = IntVar()
        self.st1.set(3)
        Entry(textvariable=self.st1, width=4, justify='center').grid(row=1, column=5, sticky="w")

        Label(text="Для X₂: ", bg="gray", font=("Comic Sans MS", 9)).grid(row=2, column=4, sticky="e")
        self.st2 = IntVar()
        self.st2.set(3)
        Entry(textvariable=self.st2, width=4, justify='center').grid(row=2, column=5, sticky="w")

        Label(text="Для X₃: ", bg="gray", font=("Comic Sans MS", 9)).grid(row=3, column=4, sticky="e")
        self.st3 = IntVar()
        self.st3.set(3)
        Entry(textvariable=self.st3, width=4, justify='center').grid(row=3, column=5, sticky="w")

        Label(text="№Y: ", bg="gray", font=("Comic Sans MS", 9)).grid(row=4, column=4, sticky="e")
        self.yvar = IntVar()
        self.yvar.set(1)
        w = OptionMenu(self.root, self.yvar, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        w.grid(row=4, column=5, sticky="w")
        w.config(bg="#696969")
        w["highlightthickness"] = 0
        w["bd"] = 2

        Label(text="Поліноми", fg="blue", bg="gray", font=('Haettenschweiler 15 underline')).grid(row=0, column=6,
                                                                                                  columnspan=1)
        self.polynom_number = IntVar()

        Radiobutton(text='Поліном Чебешова', variable=self.polynom_number, value=0, padx=25, pady=3, bg="gray",
                    font=("Comic Sans MS", 9)).grid(row=1, column=6, columnspan=1,
                                                    sticky="w")
        Radiobutton(text='Поліном Лежандра', variable=self.polynom_number, value=1, padx=25, pady=3, bg="gray",
                    font=("Comic Sans MS", 9)).grid(row=2, column=6, columnspan=1,
                                                    sticky="w")
        Radiobutton(text='Поліном Лагерра', variable=self.polynom_number, value=2, padx=25, pady=3, bg="gray",
                    font=("Comic Sans MS", 9)).grid(row=3, column=6, columnspan=1,
                                                    sticky="w")
        Radiobutton(text='Поліном Ерміта', variable=self.polynom_number, value=3, padx=25, pady=3, bg="gray",
                    font=("Comic Sans MS", 9)).grid(row=4, column=6, columnspan=1,
                                                    sticky="w")

        Label(text="Графік", fg="blue", bg="gray", font=('Haettenschweiler 15 underline')).grid(row=0, column=7,
                                                                                                columnspan=1)
        self.gr1 = BooleanVar()
        self.gr2 = BooleanVar()
        self.gr3 = BooleanVar()
        self.gr4 = BooleanVar()
        self.gr5 = BooleanVar()
        self.gr6 = BooleanVar()
        self.gr1.set(True)
        self.gr2.set(True)
        self.gr3.set(True)
        self.gr4.set(True)
        self.gr5.set(False)
        self.gr6.set(False)
        Checkbutton(text='Графік за вибіркою', variable=self.gr1, onvalue=1, offvalue=0, bg="gray",
                    font=("Comic Sans MS", 9)).grid(row=1, column=7, sticky="w")
        Checkbutton(text='Графік за координатами', variable=self.gr2, onvalue=1, offvalue=0, bg="gray",
                    font=("Comic Sans MS", 9)).grid(row=2, column=7, sticky="w")
        Checkbutton(text='Нормований вид', variable=self.gr3, onvalue=1, offvalue=0, bg="gray",
                    font=("Comic Sans MS", 9)).grid(row=3, column=7, sticky="w")
        Checkbutton(text='λ декількох систем', variable=self.gr4, onvalue=1, offvalue=0, bg="gray",
                    font=("Comic Sans MS", 9)).grid(row=4, column=7, sticky="w")
        Checkbutton(text='Вивести коефіцієнти', variable=self.gr5, onvalue=1, offvalue=0, bg="gray",
                    font=("Comic Sans MS", 9)).grid(row=4, column=0, columnspan=2, sticky="ew")

        Checkbutton(text='Вивести точковий графік', variable=self.gr6, onvalue=1, offvalue=0, bg="gray",
                    font=("Comic Sans MS", 9)).grid(row=6, column=2, columnspan=3, sticky="w")

        Button(self.root, text="Виконати", command=self.graphik, font=("Comic Sans MS", 9), activebackground="grey50",
               bg="green", bd=3).grid(row=6, column=0, columnspan=8, pady=10)
        Button(self.root, text="Очистити", command=self.clearToTextInput, font=("Comic Sans MS", 9),
               activebackground="grey50", bg="brown", bd=3).grid(row=6, column=1, columnspan=7)

        self.output = Text(wrap=WORD, width=115, height=30, font=("Comic Sans MS", 10), bg="#393636", fg="white", bd=2)
        self.output.grid(row=5, column=0, columnspan=10)
        scrollb = Scrollbar(self.root, command=self.output.yview)
        scrollb.grid(row=5, column=10, sticky='nsew')
        self.output['yscrollcommand'] = scrollb.set

        self.l = Label(text="Похибка = 0", font=("Comic Sans MS", 13), bg="gray")
        self.l.grid(row=6, column=2, columnspan=6, sticky="e")

        self.root.mainloop()

    def clearToTextInput(self):
        self.output.delete("1.0", "end")
        self.l["text"] = "Похибка = 0"

    def calc(self, x1, x2, x3, lambdas, alphas, final_coeff, phi_number):
        result = 0
        n1, n2, n3 = self.sizeX1.get(), self.sizeX2.get(), self.sizeX3.get()
        P1, P2, P3 = self.st1.get(), self.st2.get(), self.st3.get()

        polynom_number = self.polynom_number.get()
        if polynom_number == 0:
            v_polynom_in_point = star_chebyshev_polynom_in_point
            v_polynom = star_chebyshev_polynom
        elif polynom_number == 1:
            v_polynom_in_point = star_legender_polynom_in_point
            v_polynom = star_legender_polynom
        elif polynom_number == 3:
            v_polynom_in_point = hermit_polynom_in_point
            v_polynom = hermit_polynom
        else:
            v_polynom_in_point = lagger_polynom_in_point
            v_polynom = lagger_polynom

        idx = 0
        for j1 in range(n1):
            for p1 in range(P1 + 1):
                cur_coeff = alphas[phi_number][0][j1] * lambdas[phi_number][idx]
                idx += 1
                result += v_polynom_in_point(x1[j1], p1) * cur_coeff * final_coeff[phi_number][0]

        idx = n1 * (P1 + 1)
        for j2 in range(n2):
            for p2 in range(P2 + 1):
                cur_coeff = alphas[phi_number][1][j2] * lambdas[phi_number][idx]
                idx += 1
                result += v_polynom_in_point(x2[j2], p2) * cur_coeff * final_coeff[phi_number][1]

        idx = n1 * (P1 + 1) + n2 * (P2 + 1)
        for j3 in range(n3):
            for p3 in range(P3 + 1):
                cur_coeff = alphas[phi_number][2][j3] * lambdas[phi_number][idx]
                idx += 1
                result += v_polynom_in_point(x3[j3], p3) * cur_coeff * final_coeff[phi_number][2]
        return result

    def graphik(self):
        Y_norm, Y_data, X1_norm, X2_norm, X3_norm, lambdas, alphas, result = self.go()
        n1, n2, n3 = self.sizeX1.get(), self.sizeX2.get(), self.sizeX3.get()
        P1, P2, P3 = self.st1.get(), self.st2.get(), self.st3.get()

        y_number = self.yvar.get() - 1
        true_value = self.gr1.get()
        pred_value = self.gr2.get()
        norm_v = self.gr3.get()
        sample_count = self.dataSize.get()

        true_v = Y_norm[:, y_number] if norm_v else Y_data[:, y_number]
        pred_v = np.array([self.calc(X1_norm[i], X2_norm[i], X3_norm[i], lambdas, alphas, result, y_number) for i in
                           range(sample_count)])

        if not norm_v:
            pred_v = pred_v * (Y_data.max() - Y_data.min()) + (Y_data.min())

        error_value = np.linalg.norm(true_v - pred_v, ord=2)
        self.l["text"] = "Похибка = " + ("%.6f" % error_value)

        if self.gr6.get() == True:
            if (true_value):
                plt.plot(np.arange(sample_count), true_v, 'o', label='true value')
            if (pred_value):
                plt.plot(np.arange(sample_count), pred_v, 'o', label='pred value')
            plt.xlabel('sample number')
            plt.ylabel('value')
            plt.legend()
            plt.title("True and predicate values of data\n Error = %.6f" % error_value)
            plt.show()

        if (true_value):
            plt.plot(np.arange(sample_count), true_v, label='true value')

        if (pred_value):
            plt.plot(np.arange(sample_count), pred_v, label='pred value')
        plt.xlabel('sample number')
        plt.ylabel('value')
        plt.legend()
        plt.title("True and predicate values of data\n Error = %.6f" % error_value)
        plt.show()

    def go(self):

        X_data = pd.read_csv(self.dataInputX.get())
        X_data = X_data.to_numpy(dtype=np.float32)

        Y_data = pd.read_csv(self.dataInputY.get())
        Y_data = Y_data.to_numpy(dtype=np.float32)

        Y_data = Y_data[:self.dataSize.get(), :self.sizeY.get()]
        X_data = X_data[:self.dataSize.get(), :]
        n1, n2, n3 = self.sizeX1.get(), self.sizeX2.get(), self.sizeX3.get()

        X1 = X_data[:, 0: n1]
        X2 = X_data[:, n1: n1 + n2]
        X3 = X_data[:, n1 + n2: n1 + n2 + n3]

        X1_norm = norm_all_matrix(X1)
        X2_norm = norm_all_matrix(X2)
        X3_norm = norm_all_matrix(X3)
        Y_norm = norm_all_matrix(Y_data)

        P1, P2, P3 = self.st1.get(), self.st2.get(), self.st3.get()

        polynom_number = self.polynom_number.get()
        if polynom_number == 0:
            v_polynom_in_point = star_chebyshev_polynom_in_point
            v_polynom = star_chebyshev_polynom
        elif polynom_number == 1:
            v_polynom_in_point = star_legender_polynom_in_point
            v_polynom = star_legender_polynom
        elif polynom_number == 3:
            v_polynom_in_point = hermit_polynom_in_point
            v_polynom = hermit_polynom
        else:
            v_polynom_in_point = lagger_polynom_in_point
            v_polynom = lagger_polynom

        lambdas = find_lambdas([X1_norm, X2_norm, X3_norm], Y_norm, [P1, P2, P3], [n1, n2, n3], v_polynom_in_point)

        alphas = find_alphas([X1_norm, X2_norm, X3_norm], Y_norm, [P1, P2, P3], [n1, n2, n3], v_polynom_in_point,
                             lambdas)

        result = find_C([X1_norm, X2_norm, X3_norm], Y_norm, [P1, P2, P3], [n1, n2, n3], v_polynom_in_point, lambdas,
                        alphas)

        degrees = [P1, P2, P3]
        sizes = [n1, n2, n3]
        phi_count = Y_norm.shape[1]

        print_ = ""

        for phi_idx in range(phi_count):
            print_ += (
                              "Ф%d(x₁, x₂, x₃) = %.6f * Ф%d1(x₁) + %.6f * Ф%d2(x₂) + %.6f * Ф%d3(x₃)" % (
                          phi_idx + 1, result[phi_idx][0], phi_idx + 1, result[phi_idx][1], phi_idx + 1,
                          result[phi_idx][2], phi_idx + 1)) + "\n\n"
        print_ += "-" * 110
        print_ += "\n\n"

        k = 1
        i = 1
        x_iter = 0

        for num_x in [0, 1, 2]:
            iter_2 = k
            k += 1
            x_iter = x_iter + 1
            psi = " "
            costil_phi = 0
            for iter_1 in range(1, self.sizeY.get() + 1):
                j = 0
                if sizes[num_x] == n1 and iter_1 < self.sizeY.get() + 1:
                    if n1 == 1:
                        alpha = "%.3f" % (alphas[costil_phi][num_x][0])
                        psi = alpha + "* Ψ%d1(x%d1[q])" % (i, i)
                        j += 1
                    else:
                        for l in range(n1):
                            alpha = "+ %.3f" % (alphas[costil_phi][num_x][j])
                            psi = psi + alpha + " * Ψ%d%d(x%d%d[q])" % (i, l + 1, i, l + 1)
                            l += 1
                            j += 1
                elif sizes[num_x] == n2 and iter_1 < self.sizeY.get() + 1:
                    for l in range(n2):
                        alpha = "+ %.3f" % (alphas[costil_phi][num_x][j])
                        psi = psi + alpha + " * Ψ%d%d(x%d%d[q])" % (i, l + 1, i, l + 1)
                        l += 1
                        j += 1
                elif sizes[num_x] == n3 and iter_1 < self.sizeY.get() + 1:
                    for l in range(n3):
                        alpha = "+ %.3f" % (alphas[costil_phi][num_x][j])
                        psi = psi + alpha + " * Ψ%d%d(x%d%d[q])" % (i, l + 1, i, l + 1)
                        l += 1
                        j += 1
                costil_phi += 1
                answer = "Ф%d%d(x%d) =" % (iter_1, iter_2, x_iter) + psi
                psi = " "
                if iter_1 == self.sizeY.get(): answer = answer + "\n"
                print_ += answer + "\n"
            i = i + 1
        print_ += '-' * 110
        print_ += "\n\n"

        for phi_idx in range(phi_count):
            answer = "Ф%d(x₁, x₂, x₃) = \n" % (phi_idx + 1)
            for num_x in [0, 1, 2]:
                idx = 0
                for z in range(num_x):
                    idx += sizes[z] * (degrees[z] + 1)
                for j in range(sizes[num_x]):
                    for deg in range(degrees[num_x] + 1):
                        answer += "+ %.3e * T%d(x%d)" % (
                            alphas[phi_idx][num_x][j] * lambdas[phi_idx][idx] * result[phi_idx][num_x], deg,
                            (num_x + 1) * 10 + j + 1)
                        idx += 1
                    answer += "\n"
            print_ += answer + "\n"
        print_ += '-' * 110
        print_ += "\n\n"

        for phi_idx in range(phi_count):
            check = 0
            print_ += "Ф%d(x₁, x₂, x₃) = \n" % (phi_idx + 1)
            for num_x in [0, 1, 2]:
                idx = 0
                for z in range(num_x):
                    idx += sizes[z] * (degrees[z] + 1)
                for j in range(sizes[num_x]):
                    current_polynom = Polynom(0, [0])
                    for deg in range(degrees[num_x] + 1):
                        current_polynom = current_polynom.sum(v_polynom(deg).mul_for_const(
                            alphas[phi_idx][num_x][j] * lambdas[phi_idx][idx] * result[phi_idx][num_x]))
                        idx += 1

                    print_ += current_polynom.print(name='x%d' % ((num_x + 1) * 10 + j + 1)) + "\n"
            print_ += "\n"
        print_ += '-' * 110 + "\n"

        for phi_idx in range(phi_count):
            check = 0
            const_add_one_time = False
            print_ += "Ф%d(x₁, x₂, x₃) = \n" % (phi_idx + 1)
            for num_x in [0, 1, 2]:
                idx = 0
                for z in range(num_x):
                    idx += sizes[z] * (degrees[z] + 1)
                for j in range(sizes[num_x]):
                    current_polynom = Polynom(0, [0])
                    for deg in range(degrees[num_x] + 1):
                        temp_polynom = v_polynom(deg).mul_for_const(
                            alphas[phi_idx][num_x][j] * lambdas[phi_idx][idx] * result[phi_idx][num_x])
                        idx += 1
                        if num_x == 0:
                            temp_polynom = temp_polynom.substitution(1 / (X1.max() - X1.min()),
                                                                     -X1.min() / (X1.max() - X1.min()))
                        elif num_x == 1:
                            temp_polynom = temp_polynom.substitution(1 / (X2.max() - X2.min()),
                                                                     -X2.min() / (X2.max() - X2.min()))
                        else:
                            temp_polynom = temp_polynom.substitution(1 / (X3.max() - X3.min()),
                                                                     -X3.min() / (X3.max() - X3.min()))

                        current_polynom = current_polynom.sum(temp_polynom)

                    current_polynom = current_polynom.mul_for_const(Y_data.max() - Y_data.min())
                    if not const_add_one_time:
                        const_add_one_time = True
                        current_polynom = current_polynom.add_const(Y_data.min())

                    print_ += current_polynom.print(name='x%d' % ((num_x + 1) * 10 + j + 1)) + " +\n"
            print_ += "\n\n"

        self.output.insert(1.0, print_)

        print_ += "-" * 110
        print_ += "\n\n"
        print_alpha = "Вывод коэффициента α:\n\n"
        print_c = "\n\nВывод коэффициента с:\n\n"
        print_lambda = "\n\nВывод коэффициента λ:\n\n"
        file = open(self.dataOutput.get(), 'w+', encoding="utf-8")
        file.write(print_alpha + '\n'.join(map(str, alphas)) + print_c + '\n'.join(
            map(str, result)) + print_lambda + '\n'.join(map(str, lambdas)))
        file.close()

        if self.gr5.get():
            os.startfile(self.dataOutput.get())
        return Y_norm, Y_data, X1_norm, X2_norm, X3_norm, lambdas, alphas, result


if __name__ == "__main__":
    Interface()
