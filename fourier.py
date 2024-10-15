from . import arrays
import numpy as np


# TODO:add checks and unit testing
# TODO: add comments and other stuff

def dft(input_array: arrays.SQArray, rho: arrays.SQArray):
    rho = rho.normalized

    N = input_array.shape[0]
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    cos_matrix = np.cos(-2 * np.pi * (i * j) / N)
    sin_matrix = np.sin(-2 * np.pi * (i * j) / N)
    W = arrays.SQArray.from_scalar_part(sin_matrix) * rho
    W.scalar = cos_matrix
    return input_array @ W


def idft(input_array: arrays.SQArray, rho: arrays.SQArray):
    rho = rho.normalized

    N = input_array.shape[0]
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    cos_matrix = np.cos(2 * np.pi * (i * j) / N)
    sin_matrix = np.sin(2 * np.pi * (i * j) / N)
    dft_mat = arrays.SQArray.from_scalar_part(sin_matrix) * rho
    dft_mat.scalar = cos_matrix
    dft_mat /= N
    return input_array @ dft_mat


def dft2(input_array: arrays.SQArray, rho: arrays.SQArray):
    rho = rho.normalized

    n1, n2 = input_array.shape[0:2]

    # compute first fourier matrix
    i, j = np.meshgrid(np.arange(n1), np.arange(n1))
    theta = -2 * np.pi * (i * j) / n1
    cos_matrix = np.cos(theta)
    sin_matrix = np.sin(theta)
    dft_mat_1 = arrays.SQArray.from_scalar_part(sin_matrix) * rho
    dft_mat_1.scalar = cos_matrix

    # compute second fourier matrix
    i, j = np.meshgrid(np.arange(n2), np.arange(n2))
    theta = -2 * np.pi * (i * j) / n2
    cos_matrix = np.cos(theta)
    sin_matrix = np.sin(theta)
    dft_mat_2 = arrays.SQArray.from_scalar_part(sin_matrix) * rho
    dft_mat_2.scalar = cos_matrix

    output = (input_array @ dft_mat_2).T @ dft_mat_1

    return output.T


def idft2(input_array: arrays.SQArray, rho: arrays.SQArray):
    rho = rho.normalized

    n1, n2 = input_array.shape[0:2]
    # compute first fourier matrix
    i, j = np.meshgrid(np.arange(n1), np.arange(n1))
    theta = 2 * np.pi * (i * j) / n1
    cos_matrix = np.cos(theta)
    sin_matrix = np.sin(theta)
    dft_mat_1 = arrays.SQArray.from_scalar_part(sin_matrix) * rho
    dft_mat_1.scalar = cos_matrix
    dft_mat_1 /= n1

    # compute second fourier matrix
    i, j = np.meshgrid(np.arange(n2), np.arange(n2))
    theta = 2 * np.pi * (i * j) / n2
    cos_matrix = np.cos(theta)
    sin_matrix = np.sin(theta)
    dft_mat_2 = arrays.SQArray.from_scalar_part(sin_matrix) * rho
    dft_mat_2.scalar = cos_matrix
    dft_mat_2 /= n2

    output = (input_array @ dft_mat_2).T @ dft_mat_1

    return output.T
