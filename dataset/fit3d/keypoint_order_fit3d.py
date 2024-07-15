# -*- coding: utf-8 -*-
"""
Created on 07.06.24

@author: Katja

"""
from smplx.joint_names import JOINT_NAMES


class Fit3DOrder:
    names = ["pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
     "left_index",
     "left_thumb",
     "right_index",
     "right_thumb",
     "left_big_toe",
     "left_small_toe",
     "left_heel",
     "right_big_toe",
     "right_small_toe",
     "right_heel",
     "right_eye",
     "left_eye",
     "right_ear",
     "left_ear",
     "nose"
     ]

    num_joints = len(names)

    @classmethod
    def index(cls, name):
        return cls.names.index(name)

    @classmethod
    def from_SMPLX_order(cls, use_nose=True):
        indices = [JOINT_NAMES.index(name) for name in cls.names]
        if use_nose:
            return indices
        return indices[:-1]

    @classmethod
    def flip_lr_order(cls):
        res = []
        for name in cls.names:
            if "left" in name:
                new_name = name.replace("left", "right")
            elif "right" in name:
                new_name = name.replace("right", "left")
            else:
                new_name = name
            res.append(cls.index(new_name))
        return res

fit3d_reference_joints = [Fit3DOrder.index("left_hip"), Fit3DOrder.index("right_shoulder")]