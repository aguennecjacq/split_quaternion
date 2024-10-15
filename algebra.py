"""Fundamental operation/functions for the split-quaternion algebra.

The last input in the functions is used as the output. Moreover, the use of
numba's guvectorize allows the functions to integrated as numpy's u-func. In
short: this allows for the different functions in  this module to be used instead
of the usual matrix operations, e.g for two split-quaternions matrixes Q1 and Q2 we
can perform operations of the form Q1*Q2 where the multiplication is a point-wise
split-quaternion multipliplication.
"""

import numpy as np
import math
from numba import guvectorize, float64, boolean, jit, njit

# from scipy.linalg import logm


_resolution = 10 * np.finfo(float).resolution


@guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def add(q1, q2, qout):
    """Add two quaternions q1+q2"""
    qout[0] = q1[0] + q2[0]
    qout[1] = q1[1] + q2[1]
    qout[2] = q1[2] + q2[2]
    qout[3] = q1[3] + q2[3]


@guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def subtract(q1, q2, qout):
    """Subtract quaternion q1-q2"""
    qout[0] = q1[0] - q2[0]
    qout[1] = q1[1] - q2[1]
    qout[2] = q1[2] - q2[2]
    qout[3] = q1[3] - q2[3]


@guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def multiply(q1, q2, qout):
    """Multiply quaternions q1*q2"""
    qout[0] = q1[0] * q2[0] - q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]
    qout[1] = q1[0] * q2[1] + q1[1] * q2[0] - q1[2] * q2[3] + q1[3] * q2[2]
    qout[2] = q1[0] * q2[2] + q1[2] * q2[0] - q1[1] * q2[3] + q1[3] * q2[1]
    qout[3] = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]


@guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def divide(q1, q2, qout):
    """Divide quaternions q1/q2 = q1 * q2.inverse"""
    q2norm = q2[0] ** 2 + q2[1] ** 2 - q2[2] ** 2 - q2[3] ** 2
    qout[0] = (q1[0] * q2[0] + q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]) / q2norm
    qout[1] = (-q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]) / q2norm
    qout[2] = (-q1[0] * q2[2] + q1[2] * q2[0] + q1[1] * q2[3] - q1[3] * q2[1]) / q2norm
    qout[3] = (-q1[0] * q2[3] + q1[3] * q2[0] - q1[1] * q2[2] + q1[2] * q2[1]) / q2norm


@guvectorize([(float64, float64[:], float64[:])], '(),(n)->(n)')
def multiply_scalar(s, q, qout):
    """Multiply scalar by quaternion s*q"""
    qout[0] = s * q[0]
    qout[1] = s * q[1]
    qout[2] = s * q[2]
    qout[3] = s * q[3]


@guvectorize([(float64, float64[:], float64[:])], '(),(n)->(n)')
def divide_scalar(s, q, qout):
    """Divide scalar by quaternion s/q = s * q.inverse"""
    qnorm = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    qout[0] = s * q[0] / qnorm
    qout[1] = -s * q[1] / qnorm
    qout[2] = -s * q[2] / qnorm
    qout[3] = -s * q[3] / qnorm


@guvectorize([(float64[:], float64, float64[:])], '(n),()->(n)')
def scalar_multiply(q, s, qout):
    """Multiply quaternion by scalar q*s"""
    qout[0] = q[0] * s
    qout[1] = q[1] * s
    qout[2] = q[2] * s
    qout[3] = q[3] * s


@guvectorize([(float64[:], float64, float64[:])], '(n),()->(n)')
def scalar_divide(q, s, qout):
    """Divide quaternion by scalar q/s"""
    qout[0] = q[0] / s
    qout[1] = q[1] / s
    qout[2] = q[2] / s
    qout[3] = q[3] / s


@guvectorize([(float64[:], float64[:])], '(n)->(n)')
def negative(q, qout):
    """Return negative quaternion -q"""
    qout[0] = -q[0]
    qout[1] = -q[1]
    qout[2] = -q[2]
    qout[3] = -q[3]


@guvectorize([(float64[:], float64[:])], '(n)->(n)')
def positive(q, qout):
    """Return input quaternion q"""
    qout[0] = q[0]
    qout[1] = q[1]
    qout[2] = q[2]
    qout[3] = q[3]


@guvectorize([(float64[:], float64[:])], '(n)->()')
def absolute(q, qout):
    """Return absolute norm value of split quaternion q"""
    qout[0] = math.sqrt(abs(q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2))


@guvectorize([(float64[:], float64[:])], '(n)->(n)')
def conj(q, qout):
    """Return quaternion-conjugate of quaternion q̄"""
    qout[0] = +q[0]
    qout[1] = -q[1]
    qout[2] = -q[2]
    qout[3] = -q[3]


@guvectorize([(float64[:], float64[:])], '(n)->(n)')
def exp(q, qout):
    """Return exponential of input quaternion exp(q)"""
    vnorm = q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    e = np.exp(q[0])
    if vnorm > _resolution:
        vnorm = math.sqrt(vnorm)
        s = np.sin(vnorm) / vnorm
        qout[0] = e * np.cos(vnorm)
        qout[1] = e * s * q[1]
        qout[2] = e * s * q[2]
        qout[3] = e * s * q[3]
    elif vnorm < -_resolution:
        vnorm = math.sqrt(-vnorm)
        s = np.sinh(vnorm) / vnorm
        qout[0] = e * np.cosh(vnorm)
        qout[1] = e * s * q[1]
        qout[2] = e * s * q[2]
        qout[3] = e * s * q[3]
    else:
        qout[0] = np.exp(q[0])
        qout[1] = e * q[1]
        qout[2] = e * q[2]
        qout[3] = e * q[3]


# We leve log out for the time being
# TODO ?
# def log(q, qout):
#     """
#         Return logarithm of input split quaternion log(q).
#         At the moment, we bypass the direct computations and use the matrix log functions.
#     """
#     q_mat = np.array([[q[0] + q[2], q[3] - q[1]], [q[3] + q[1], q[0] - q[2]]])
#     q_mat = logm(q_mat)
#     if q_mat.imag.sum() > _resolution:
#         qout[0] = math.nan
#         qout[1] = math.nan
#         qout[2] = math.nan
#         qout[3] = math.nan
#     else:
#         qout[0] = (q_mat[0, 0] + q_mat[1, 1]) / 2
#         qout[1] = (q_mat[1, 0] - q_mat[0, 1]) / 2
#         qout[2] = (q_mat[0, 0] - q_mat[1, 1]) / 2
#         qout[3] = (q_mat[1, 0] + q_mat[0, 1]) / 2


@guvectorize([(float64[:], float64[:])], '(n)->(n)')
def sqrt(q, qout):
    """Return square-root of input split quaternion √q.
    References: The roots of a split quaternion - M. Özdemir
    """
    qnorm = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    vnorm = q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    vabs = q[1] ** 2 + q[2] ** 2 + q[3] ** 2

    # null vector part
    if vabs < _resolution:
        qout[0] = math.sqrt(q[0])
        qout[1] = 0
        qout[2] = 0
        qout[3] = 0
    # time like vector part
    elif vnorm > _resolution:
        qnorm = math.sqrt(qnorm)
        denominator = math.sqrt(2 * (q[0] + qnorm))
        qout[0] = math.sqrt((q[0] + qnorm) / 2)
        qout[1] = q[1] / denominator
        qout[2] = q[2] / denominator
        qout[3] = q[3] / denominator

    # time like, the scalar part must be non-negative
    elif qnorm > _resolution and q[0] >= 0:
        qnorm = math.sqrt(qnorm)
        denominator = math.sqrt(2 * (q[0] + qnorm))
        qout[0] = math.sqrt((q[0] + qnorm) / 2)
        qout[1] = q[1] / denominator
        qout[2] = q[2] / denominator
        qout[3] = q[3] / denominator

    # light like vector part
    elif abs(vnorm) < _resolution and q[0] > _resolution:
        denominator = 2 * math.sqrt(q[0])
        qout[0] = math.sqrt(q[0])
        qout[1] = q[1] / denominator
        qout[2] = q[2] / denominator
        qout[3] = q[3] / denominator

    # light like sq
    elif abs(qnorm) < _resolution and q[0] > _resolution:
        denominator = math.sqrt(2 * q[0])
        qout[0] = math.sqrt(q[0] / 2)
        qout[1] = q[1] / denominator
        qout[2] = q[2] / denominator
        qout[3] = q[3] / denominator
    else:
        # No possible square root in this case
        # warnings.warn("invalid value encountered in sqrt", RuntimeWarning)
        qout[0] = math.nan
        qout[1] = math.nan
        qout[2] = math.nan
        qout[3] = math.nan


@guvectorize([(float64[:], float64[:])], '(n)->(n)')
def square(q, qout):
    """Return square of quaternion q*q"""
    qout[0] = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 + q[3] ** 2
    qout[1] = 2 * q[0] * q[1]
    qout[2] = 2 * q[0] * q[2]
    qout[3] = 2 * q[0] * q[3]


@guvectorize([(float64[:], float64[:])], '(n)->(n)')
def reciprocal(q, qout):
    """Return reciprocal (inverse) of the split quaternion q.inverse"""
    norm = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    if abs(norm) > _resolution:
        qout[0] = q[0] / norm
        qout[1] = -q[1] / norm
        qout[2] = -q[2] / norm
        qout[3] = -q[3] / norm
    else:
        qout[0] = math.nan
        qout[1] = math.nan
        qout[2] = math.nan
        qout[3] = math.nan


@guvectorize([(float64[:], float64[:])], '(n)->()')
def angle(q, qout):
    """
    Returns the angle (in radians) of a split quaternion with time-like vector part.
    For q = |q|.exp(θ.ρ)', this returns θ` ∈[-2π,2π]. If the vector part of the split quaternion
    is not timelike, then this returns nan

    """
    vnorm = q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    if vnorm > _resolution:
        vnorm = math.sqrt(vnorm)
        qout[0] = np.arctan2(q[0], vnorm)
    else:
        qout[0] = math.nan


@guvectorize([(float64[:], float64[:], boolean[:])], '(n),(n)->()')
def not_equal(q1, q2, bout):
    bout[0] = np.any(q1[:] != q2[:])


@guvectorize([(float64[:], float64[:], boolean[:])], '(n),(n)->()')
def equal(q1, q2, bout):
    bout[0] = np.all(q1[:] == q2[:])


@guvectorize([(float64[:], float64[:], boolean[:])], '(n),(n)->()')
def logical_and(q1, q2, bout):
    bout[0] = np.any(q1[:]) and np.any(q2[:])


@guvectorize([(float64[:], float64[:], boolean[:])], '(n),(n)->()')
def logical_or(q1, q2, bout):
    bout[0] = np.any(q1[:]) or np.any(q2[:])


@guvectorize([(float64[:], boolean[:])], '(n)->()')
def isfinite(qin, bout):
    bout[0] = np.isfinite(qin[0]) and np.isfinite(qin[1]) and np.isfinite(qin[2]) and np.isfinite(qin[3])


@guvectorize([(float64[:], boolean[:])], '(n)->()')
def isinf(qin, bout):
    bout[0] = np.isinf(qin[0]) or np.isinf(qin[1]) or np.isinf(qin[2]) or np.isinf(qin[3])


@guvectorize([(float64[:], boolean[:])], '(n)->()')
def isnan(qin, bout):
    bout[0] = np.isnan(qin[0]) or np.isnan(qin[1]) or np.isnan(qin[2]) or np.isnan(qin[3])


@jit([(float64[:, :, :], float64[:, :, :], float64[:, :, :])], nopython=True)
def matmul(q1, q2, qout):
    q1_0 = np.ascontiguousarray(q1[..., 0])
    q1_1 = np.ascontiguousarray(q1[..., 1])
    q1_2 = np.ascontiguousarray(q1[..., 2])
    q1_3 = np.ascontiguousarray(q1[..., 3])

    q2_0 = np.ascontiguousarray(q2[..., 0])
    q2_1 = np.ascontiguousarray(q2[..., 1])
    q2_2 = np.ascontiguousarray(q2[..., 2])
    q2_3 = np.ascontiguousarray(q2[..., 3])

    qout[..., 0] = q1_0 @ q2_0 - q1_1 @ q2_1 + q1_2 @ q2_2 + q1_3 @ q2_3
    qout[..., 1] = q1_0 @ q2_1 + q1_1 @ q2_0 - q1_2 @ q2_3 + q1_3 @ q2_2
    qout[..., 2] = q1_0 @ q2_2 + q1_2 @ q2_0 - q1_1 @ q2_3 + q1_3 @ q2_1
    qout[..., 3] = q1_0 @ q2_3 + q1_3 @ q2_0 + q1_1 @ q2_2 - q1_2 @ q2_1


# Aliases
conjugate = conj
scalar_true_divide = scalar_divide
true_divide_scalar = divide_scalar
true_divide = divide
