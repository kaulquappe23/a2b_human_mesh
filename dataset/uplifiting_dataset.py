# Code adapted from: https://github.com/facebookresearch/VideoPose3D
# Original Code: Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
# Current Code: Copyright (c) 2022-present, Machine Learning and Computer Vision Lab, University of Augsburg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# author:  goldbricklemon


import torch


def world_to_cam_and_2d(sequence_3d, cam):
    """
    Transform pose sequence in world coords to cam coords and project to 2D
    :param sequence_3d: (N, K, 3)
    :param cam: (4+3+11) rot/trans/intrinsics
    :param mask:(N)
    :param stride_mask:(N)
    :return:(N,K,3), (N,K,2) mask, stride_mask
    """
    quat_rot = cam[:4]
    trans = cam[4:7]
    intrinsics = cam[7:19]
    # res = cam[7:9]
    # focal_length = cam[9:11]
    # center = cam[11:13]
    # radial_dist = cam[13:16]
    # tang_dist = cam[16:19]

    # pt_world_to_cam
    sequence_3d_cam = pt_world_to_cam(sequence_3d, R=quat_rot, t=trans)
    # cam_to_2d
    sequence_2d = pt_project_to_2d(sequence_3d_cam, intrinsics=intrinsics)
    # done
    return sequence_3d_cam, sequence_2d


def pt_qrot(q, v):
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.linalg.cross(qvec, v)
    uuv = torch.linalg.cross(qvec, uv)
    return v + 2 * (q[..., :1] * uv + uuv)


def pt_qinverse(q):
    # Quaternion must be normalized
    w = q[..., :1]
    xyz = q[..., 1:]
    return torch.concat([w, -xyz], dim=-1)


def pt_world_to_cam(X, R, t):
    Rt = pt_qinverse(R)  # Invert rotation
    Rt = torch.reshape(Rt, shape=(1,) * len(X.shape[:-1]) + Rt.shape)
    intermediate = torch.tile(Rt, dims=(*X.shape[:-1], 1))
    return pt_qrot(intermediate, X - t)  # Rotate and translate


def pt_project_to_2d_linear(X, intrinsics):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, K, 3)
    camera_params -- intrinsic parameters (2+2+2+3+2=11)
    """
    intrinsics = torch.reshape(intrinsics, shape=(1, 1, -1))

    f = intrinsics[..., 2:4]
    c = intrinsics[..., 4:6]
    XX = X[..., :2] / X[..., 2:]
    XX = torch.minimum(torch.maximum(XX, -1.), 1.)

    return f * XX + c


def pt_project_to_2d(X, intrinsics):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.

    Arguments:
    X -- 3D points in *camera space* to transform (N, K, 3)
    camera_params -- intrinsic parameters (2+2+2+3+2=11)
    """
    intrinsics = torch.reshape(intrinsics, shape=(1, 1, -1))

    f = intrinsics[..., 2:4]
    c = intrinsics[..., 4:6]
    k = intrinsics[..., 6:9]
    p = intrinsics[..., 9:]

    XX = X[..., :2] / X[..., 2:]
    XX = torch.minimum(torch.maximum(XX, torch.Tensor((-1.,))), torch.Tensor((1.,)))
    r2 = torch.sum(XX[..., :2] ** 2, dim=-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.concat([r2, r2 ** 2, r2 ** 3], dim=-1), dim=-1, keepdim=True)
    tan = torch.sum(p * XX, dim=-1, keepdim=True)

    XXX = XX * (radial + tan) + p * r2

    return f * XXX + c
