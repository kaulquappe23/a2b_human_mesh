# -*- coding: utf-8 -*-
"""
Created on 09.08.24

@author: Katja

"""
import os

from aspset510 import Aspset510, Camera

from dataset.aspset.keypoint_order import APSSet17POrder


def load_aspset_gt(data_path, split):
    gt = {}
    aspset = Aspset510(data_path)
    for s, c in aspset.splits[split]:
        if s not in gt:
            gt[s] = {}
        if c not in gt[s]:
            gt[s][c] = {}
        clip = aspset.clip(s, c)
        for cam in clip.camera_ids:
            camera = clip.load_camera(cam)
            mocap = clip.load_mocap()
            joints_3d = mocap.joint_positions
            extrinsic = camera.extrinsic_matrix.copy()
            # convert_mm_to_m:
            joints_3d = joints_3d / 1000
            extrinsic[0:3, 3] = extrinsic[0:3, 3] / 1000  # mm to meters
            camera = Camera(camera.intrinsic_matrix, extrinsic)
            joints_3d_cam = camera.world_to_camera_space(joints_3d.copy())
            joints_3d_cam = joints_3d_cam[:, APSSet17POrder.to_our_17p_order()]
            gt[s][c][cam] = joints_3d_cam

    return gt