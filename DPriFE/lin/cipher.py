import secrets
from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from DPriFE.utils.rng import create_rng
from sympy import Matrix
from sympy.core.numbers import Rational


def header(name):
    print("======", name, "======")


PublicParameter = namedtuple(
    "PublicParameter",
    [
        "l",  # int  # input size
        "m",  # int  # protocol parameter
        "n",  # int  # protocol parameter
        "p_1",  # int  # upper bound of the inner product
        "p_2",  # int  # random parameter
        "alpha",  # float  # random parameter
        "lbda",  # int  # upper bound of the noise
    ],
)


# @dataclass
# class PublicParameter:
#     l: int  # input size
#     m: int  # protocol parameter
#     n: int  # protocol parameter
#     p_1: int  # upper bound of the inner product
#     p_2: int  # random parameter
#     alpha: float  # random parameter
#     lbda: int  # upper bound of the noise


@dataclass
class PK:
    W: np.matrix  # m x n
    W_inv: np.matrix  # m x n
    A: np.matrix  # (l+2) x n


@dataclass
class MSK:
    M: np.matrix  # m x (l+2)
    E: np.matrix  # (l+2) x n


@dataclass
class CT:
    c_0: np.array  # m
    c_1: np.array  # l+x


@dataclass
class SK:
    k: np.array  # m
    y_hat: np.array  # l+2


class Simluator:
    param: PublicParameter
    rng = create_rng(secrets.randbits(256))

    def __init__(self, param: PublicParameter) -> None:
        self.param = param

    def setup(self) -> Tuple[PK, MSK]:
        while True:
            try:
                W: np.matrix = self.rng.integers(0, self.param.p_2, (self.param.m, self.param.n))
                # W = Matrix([[1, -2, 3], [0, -1, 4], [0, 0, 1]])
                W = Matrix([[1, 0, 0, 0], [3, -1, 0, 0], [0, 1, -1, 0], [2, 4, 3, 1]])
                W_inv = Matrix(W).inv()
            except np.linalg.LinAlgError:
                continue
            break

        M: np.matrix = self.rng.integers(0, self.param.p_2, (self.param.m, self.param.l + 2))
        E: np.matrix = self.rng.normal(0, self.param.alpha, (self.param.l + 2, self.param.n)).round(0).astype(int)

        A = M.transpose() @ W + E

        # header("W")
        # print(W)
        # header("W_inv")
        # print(W_inv)
        # header("M")
        # print(M)
        # header("E")
        # print(E)
        # header("A")
        # print(A)

        return PK(W=W, W_inv=W_inv, A=A), MSK(M, E)

    def enc(self, pk: PK, x: np.array) -> CT:
        s: np.array = self.rng.integers(0, self.param.p_2, self.param.n)
        r: int = self.rng.integers(0, self.param.p_1)
        e_0: np.array = self.rng.normal(0, self.param.alpha, self.param.m).round(0).astype(int)
        e_1: np.array = self.rng.normal(0, self.param.alpha, self.param.l + 2).round(0).astype(int)
        x_hat: np.array = np.append(x, [-r, 1])

        c_0 = pk.W @ s + self.param.lbda * e_0
        c_1 = pk.A @ s + self.param.lbda * e_1 + x_hat

        # header("s")
        # print(s)
        # header("r")
        # print(r)
        # header("e_0")
        # print(e_0)
        # header("e_1")
        # print(e_1)
        # header("x_hat")
        # print(x_hat)
        # header("c_0")
        # print(c_0)
        # header("c_1")
        # print(c_1)

        return CT(c_0=c_0, c_1=c_1), r

    def key_gen(self, pk: PK, msk: MSK, y: np.array, r: int, sigma: int) -> SK:
        e = round(self.rng.normal(0, sigma))
        y_hat = np.append(y, [1, e + r])
        k = (msk.M + (msk.E @ pk.W_inv).transpose()) @ y_hat

        # header("e")
        # print(e)
        # header("y_hat")
        # print(y_hat)
        # header("k")
        # print(k)

        return SK(k=k, y_hat=y_hat)

    def dec(self, sk: SK, ct: CT):
        res = np.inner(ct.c_1, sk.y_hat) - np.inner(ct.c_0, sk.k)
        return res % self.param.lbda
