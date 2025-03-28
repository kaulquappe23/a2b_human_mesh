# -*- coding: utf-8 -*-
"""
Created on 16 Dec 2022, 16:05

Code adapted from: https://github.com/acvictor/DLT


"""

import numpy as np
import cv2


def dlt_input_normalization(nd, x):
    '''
    Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).
    Input
    -----
    nd: number of dimensions, 3 here
    x: the data to be normalized (directions at different columns and points at rows)
    Output
    ------
    Tr: the transformation matrix (translation plus scaling)
    x: the transformed data
    '''

    x = np.asarray(x)
    m, s = np.mean(x, 0), np.std(x)
    if nd == 2:
        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])

    Tr = np.linalg.inv(Tr)
    x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
    x = x[0:nd, :].T

    return Tr, x


def dlt_calib(nd, xyz, uv):
    '''
    Camera calibration by DLT using known object points and their image points.
    Input
    -----
    nd: dimensions of the object space, 3 here.
    xyz: coordinates in the object 3D space.
    uv: coordinates in the image 2D space.
    The coordinates (x,y,z and u,v) are given as columns and the different points as rows.
    There must be at least 6 calibration points for the 3D DLT.
    Output
    ------
     L: array of 11 parameters of the calibration matrix.
     err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
    '''
    if (nd != 3):
        raise ValueError('%dD DLT unsupported.' % (nd))

    # Converting all variables to numpy array
    xyz = np.asarray(xyz)
    uv = np.asarray(uv)

    n = xyz.shape[0]

    # Validating the parameters:
    if uv.shape[0] != n:
        raise ValueError('Object (%d points) and image (%d points) have different number of points.' % (n, uv.shape[0]))

    if (xyz.shape[1] != 3):
        raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).' % (xyz.shape[1], nd, nd))

    if (n < 6):
        raise ValueError(
            '%dD DLT requires at least %d calibration points. Only %d points were entered.' % (nd, 2 * nd, n))

    # Normalize the data to improve the DLT quality (DLT is dependent of the system of coordinates).
    # This is relevant when there is a considerable perspective distortion.
    # Normalization: mean position at origin and mean distance equals to 1 at each direction.
    Txyz, xyzn = dlt_input_normalization(nd, xyz)
    Tuv, uvn = dlt_input_normalization(2, uv)

    A = []

    for i in range(n):
        x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
        u, v = uvn[i, 0], uvn[i, 1]
        A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])

    # Convert A to array
    A = np.asarray(A)

    # Find the 11 parameters:
    U, S, V = np.linalg.svd(A)

    # The parameters are in the last line of Vh and normalize them
    L = V[-1, :] / V[-1, -1]
    # print(L)
    # Camera projection matrix
    P = L.reshape(3, nd + 1)
    # print(P)

    # Denormalization
    # pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
    P = np.dot(np.dot(np.linalg.pinv(Tuv), P), Txyz)
    # print(P)
    P = P / P[-1, -1]
    #print(np.round(P, 2))

    # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
    uv2 = np.dot(P, np.concatenate((xyz.T, np.ones((1, xyz.shape[0])))))
    uv2 = uv2 / uv2[2, :]
    # Mean distance:
    err = np.sqrt(np.mean(np.sum((uv2[0:2, :].T - uv) ** 2, 1)))

    # # Decompose projection matrix to K, R, t
    # # See: https://www.ipb.uni-bonn.de/html/teaching/msr2-2020/sse2-13-DLT.pdf
    # H = P[:, 0:3]
    # h = P[:, 3:4]
    # H_inv = np.linalg.inv(H)
    # R_T, K_inv = np.linalg.qr(H_inv)
    # R = R_T.T
    # K = np.linalg.inv(K_inv)
    # C = (- H_inv) @ h
    # t = (-R) @ C

    ret = cv2.decomposeProjectionMatrix(P)
    K, R, t = ret[:3]
    K = K / K[-1, -1]
    t = t / t[-1]
    # Convert from R(P - t) to RP + t
    t = np.matmul(R, t[:3,:]) * -1.
    return P, K, R, t, err


def test_dlt():
    # Known 3D coordinates
    xyz = [[-875, 0, 9.755], [442, 0, 9.755], [1921, 0, 9.755], [2951, 0.5, 9.755], [-4132, 0.5, 23.618],
           [-876, 0, 23.618]]
    # Known pixel coordinates
    uv = [[76, 706], [702, 706], [1440, 706], [1867, 706], [264, 523], [625, 523]]

    nd = 3
    P, K, R, t, err = dlt_calib(nd, xyz, uv)
    print('Matrix')
    print(P)
    print('\nError')
    print(err)


if __name__ == "__main__":
    test_dlt()