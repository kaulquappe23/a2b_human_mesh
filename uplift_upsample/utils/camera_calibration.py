'''
Find the extrinsic and instrinsic camera matrices given many measurements in image coordinates and their corresponding
3D coordinates.
'''
import random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from uplift_upsample.utils.dlt import dlt_calib

np.random.seed(42)
random.seed(42)

# convert euclidean to homogenous coordinates
def to_homogenous(point):
    if len(point.shape) == 1:
        return np.hstack((point, np.ones((1,))))
    else:
        return np.hstack((point, np.ones((point.shape[0], 1))))

#convert homogenous to euclidean coordinates
def to_euclidean(point):
    return point[..., :-1] / np.expand_dims(point[..., -1], -1)

#adds random noise vector to vector
def add_noise(vector):
    return vector + np.random.normal(0, 0.005, vector.shape)


# regression to find the camera matrices
def regress_cameramatrices(points2d, points3d, H_image, W_image, startmatrices=None):
    '''
    points2d: [(key, point2d), (key, point2d), ...]
    startmatrices: (Mint, Mext)
    return: (Min, Mext)
    '''
    #error function for optimization
    def summed_distance(points2d, points3d, Mint, Mext):
        projected_points = to_euclidean(np.einsum('i j, j k , b k -> b i', Mint, Mext, to_homogenous(points3d)))
        distance = np.sum(np.sqrt(np.sum(np.square(projected_points - points2d), axis=1)))
        distance /= len(points2d)
        return distance

    #function to be optimized
    def opt_func(x):
        px = W_image // 2
        py = H_image // 2
        fx, fy, tx, ty, tz, a, b, c = x
        Mint = np.array([[fx, 0, px, 0],
                         [0, fy, py, 0],
                         [0, 0, 1, 0], ])

        rot = R.from_euler('xyz', [a, b, c], degrees=False).as_matrix()
        Mext = np.array([
            [rot[0, 0], rot[0, 1], rot[0, 2], tx],
            [rot[1, 0], rot[1, 1], rot[1, 2], ty],
            [rot[2, 0], rot[2, 1], rot[2, 2], tz],
            [0, 0, 0, 1]
        ])
        return summed_distance(points2d_lst, points3d_lst, Mint, Mext)

    points2d_lst, points3d_lst = [], []
    for key, point in points2d:
        points2d_lst.append(point)
        points3d_lst.append(points3d[key])
    points3d_lst = np.array(points3d_lst)

    #initial guess
    if startmatrices is None:
        x0 = np.random.normal(0, 1, 8)
    else:
        Mint, Mext = startmatrices
        angles = R.from_matrix(Mext[:3, :3]).as_euler('xyz', degrees=False)
        x0 = np.array([Mint[0, 0], Mint[1, 1], Mext[0, 3], Mext[1, 3], Mext[2, 3], angles[0], angles[1], angles[2]])
    #map angles to [-pi, pi]
    x0[5:] = np.mod((x0[5:] + np.pi), (2 * np.pi)) - np.pi

    print('initial error: ', opt_func(x0))
    #bounds = [(0, None), (0, None), (None, None), (None, None), (None, None), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    #minimization to find sthe camera parameters
    res_2 = minimize(opt_func, x0, method='BFGS')# method='L-BFGS-B', bounds=bounds)

    print('final error: ', opt_func(res_2.x))

    fx, fy, tx, ty, tz, a, b, c = res_2.x

    px = W_image // 2
    py = H_image // 2

    Mint = np.array([[fx, 0, px, 0],
                     [0, fy, py, 0],
                     [0, 0, 1, 0], ])
    rot = R.from_euler('xyz', [a, b, c], degrees=False).as_matrix()
    Mext = np.array([
        [rot[0, 0], rot[0, 1], rot[0, 2], tx],
        [rot[1, 0], rot[1, 1], rot[1, 2], ty],
        [rot[2, 0], rot[2, 1], rot[2, 2], tz],
        [0, 0, 0, 1]
    ])
    return Mint, Mext


# regression to find the extrinsic matrix
def regress_extrinsic_matrix(points2d, points3d, startmatrices=None):
    '''
    points2d: [(key, point2d), (key, point2d), ...]
    startmatrices: (Mint, Mext)
    return: Mext
    '''
    #error function for optimization
    def summed_distance(points2d, points3d, Mint, Mext):
        projected_points = to_euclidean(np.einsum('i j, j k , b k -> b i', Mint, Mext, to_homogenous(points3d)))
        distance = np.sum(np.sqrt(np.sum(np.square(projected_points - points2d), axis=1)))
        distance /= len(points2d)
        return distance

    #function to be optimized
    def opt_func(x):
        tx, ty, tz, a, b, c = x

        rot = R.from_euler('xyz', [a, b, c], degrees=False).as_matrix()
        Mext = np.array([
            [rot[0, 0], rot[0, 1], rot[0, 2], tx],
            [rot[1, 0], rot[1, 1], rot[1, 2], ty],
            [rot[2, 0], rot[2, 1], rot[2, 2], tz],
            [0, 0, 0, 1]
        ])
        return summed_distance(points2d_lst, points3d_lst, Mint, Mext)

    points2d_lst, points3d_lst = [], []
    for key, point in points2d:
        points2d_lst.append(point)
        points3d_lst.append(points3d[key])
    points3d_lst = np.array(points3d_lst)

    #initial guess
    if startmatrices is None:
        raise ValueError('startmatrices must be provided')

    Mint, Mext = startmatrices
    angles = R.from_matrix(Mext[:3, :3]).as_euler('xyz', degrees=False)
    x0 = np.array([Mext[0, 3], Mext[1, 3], Mext[2, 3], angles[0], angles[1], angles[2]])
    #map angles to [-pi, pi]
    x0[5:] = np.mod((x0[5:] + np.pi), (2 * np.pi)) - np.pi

    print('initial error: ', opt_func(x0))
    #bounds = [(0, None), (0, None), (None, None), (None, None), (None, None), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    #minimization to find sthe camera parameters
    res_2 = minimize(opt_func, x0, method='BFGS')# method='L-BFGS-B', bounds=bounds)

    print('final error: ', opt_func(res_2.x))

    tx, ty, tz, a, b, c = res_2.x

    rot = R.from_euler('xyz', [a, b, c], degrees=False).as_matrix()
    Mext = np.array([
        [rot[0, 0], rot[0, 1], rot[0, 2], tx],
        [rot[1, 0], rot[1, 1], rot[1, 2], ty],
        [rot[2, 0], rot[2, 1], rot[2, 2], tz],
        [0, 0, 0, 1]
    ])
    return Mext


def DLT(points2d, points3d):
    #Direct linear transformation
    points2d_dlt, points3d_dlt = [], []
    for key, point in points2d:
        points2d_dlt.append(point)
        points3d_dlt.append(points3d[key])
    points2d_dlt = np.array(points2d_dlt)
    points3d_dlt = np.array(points3d_dlt)
    P, K, R, t, err = dlt_calib(3, points3d_dlt, points2d_dlt)
    Mext = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
    Mint = np.concatenate((K, np.array([[0, 0, 0]]).T), axis=1)
    return Mint, Mext


def calc_cameramatrices(keypoints_dict, points3d, image_height, image_width, params=None):
    '''First do a DLT to get a rough estimate of the camera matrices.
    Then use the regression to get a better estimate of the camera matrices.
    -----
    keypoints_dict: {1:[(x, y), (x, y)], 2:[(x, y), (x, y)], ...}
    params: dict containing the intrinsic parameter; if None, the intrinsic parameters are regressed too
    '''
    #Direct linear transformation
    keys_dlt = keypoints_dict.keys()
    assert len(keys_dlt) >= 6, 'not enough points for DLT'
    # points2d_lst: [(key1, point1), (key2, point2), ...]
    points2d_lst = []
    for key, points in keypoints_dict.items():
        for point in points:
            points2d_lst.append((key, point))

    # initial guess using DLT and a subset of the points
    Mint, Mext = DLT(points2d_lst, points3d)

    if params is None:
        # intrinsic matirx is calculated
        # regression using all points
        Mint, Mext = regress_cameramatrices(points2d_lst, points3d, image_height, image_width, startmatrices=(Mint, Mext))
    else:
        fx, fy, cx, cy = params['fx'], params['fy'], params['cx'], params['cy']
        Mint = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float64)
        Mext = regress_extrinsic_matrix(points2d_lst, points3d, startmatrices=(Mint, Mext))

    # with np.printoptions(precision=1, suppress=True):
    #     print('estimated Mint: \n ', Mint)
    #     print('estimated Mext: \n', Mext)
    #     print('estimated camera matrix: \n', Mint @ Mext)
    # print('-' * 50)

    return Mint, Mext


def test_on_video():
    import pandas as pd

    H_image, W_image = 720, 1280

    points3d = {
            1:  np.array([0.105, 0.1125, 0.]),
            2:  np.array([0.695, 0.1125, 0.]),
            3:  np.array([0.695, 0.4875, 0.]),
            4:  np.array([0., 0., 0.34]),
            5:  np.array([0.8, 0., 0.34]),
            6:  np.array([0.8, 0.6, 0.34]),
            7:  np.array([0., 0.6, 0.34]),
            8:  np.array([0.4, 0.3, 0.34]),
            9:  np.array([0., 0., 0.8]),
            10: np.array([0., 0.6, 0.8]),
            11: np.array([0.8, 0.6, 0.8]),
            }

    videopath = 'cut.mp4'
    keypoint_path = '../cut_keypoints.csv'
    keypoints_csv = pd.read_csv(keypoint_path, sep=';', skiprows=[0])

    keypoints_dict = {}
    for k in keypoints_csv.iterrows():
        for keypoint in k[1].keys():
            if 'flag' in keypoint and 'ball' not in keypoint and 2.01 > k[1][keypoint] > 1.99:
                key = int(keypoint.replace('_flag', ''))
                if key not in keypoints_dict:
                    keypoints_dict[key] = []
                coords = (k[1][keypoint.replace('flag', 'x')], k[1][keypoint.replace('flag', 'y')])
                keypoints_dict[key].append(coords)
        Mint, Mext = calc_cameramatrices(keypoints_dict)
        #print('intrinsics: \n', Mint)
        print(f'fx: {Mint[0, 0]}, fy: {Mint[1, 1]}')
        print('extrinsics: \n', Mext)
        print('-' * 50)

def keypoint_regression(points_3d, points_2d, im_height, im_width):
    assert points_3d.shape[0] == points_2d.shape[0]

    pt_dict_3d, pt_dict_2d = {}, {}
    for i in range(points_2d.shape[0]):
        pt_dict_3d[i] = points_3d[i]
        pt_dict_2d[i] = [points_2d[i]]

    Mint, Mext = calc_cameramatrices(pt_dict_2d, pt_dict_3d, im_height, im_width)
    print('intrinsics: \n', Mint)
    print(f'fx: {Mint[0, 0]}, fy: {Mint[1, 1]}')
    print('extrinsics: \n', Mext)
    print('-' * 50)

if __name__ == '__main__':
    test_on_video()