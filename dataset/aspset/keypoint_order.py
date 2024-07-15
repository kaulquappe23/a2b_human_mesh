# -*- coding: utf-8 -*-
"""
Created on 15 Jun 2022, 10:53

@author: einfalmo
"""
import numpy as np


class APSSet17POrder:
    """
    Helper for the original ASPSet 17-point pose definition in its original order
    """
    r_ankle = 0
    r_knee = 1
    r_hip = 2
    r_wrist = 3
    r_elbow = 4
    r_shoulder = 5
    l_ankle = 6
    l_knee = 7
    l_hip = 8
    l_wrist = 9
    l_elbow = 10
    l_shoulder = 11
    head_top = 12
    head = 13
    neck = 14
    torso = 15
    pelvis = 16

    num_points = 17

    _indices = [r_ankle, r_knee, r_hip,
                r_wrist, r_elbow, r_shoulder,
                l_ankle, l_knee, l_hip,
                l_wrist, l_elbow, l_shoulder,
                head_top, head, neck,
                torso, pelvis,
                ]

    _names = ["rank", "rknee", "rhip",
              "rwri", "relb", "rsho",
              "lank", "lknee", "lhip",
              "lwri", "lelb", "lsho",
              "htop", "head", "neck", "torso",
              "pelv",
              ]

    @classmethod
    def indices(cls):
        return cls._indices


    @classmethod
    def to_our_17p_order(cls):
        """
        Matches our 17 point order
        :return:
        """
        return [cls.r_ankle, cls.r_knee, cls.r_hip,
                cls.l_hip, cls.l_knee, cls.l_ankle,
                cls.pelvis,
                cls.neck, cls.torso,
                cls.head, cls.head_top,
                cls.r_wrist, cls.r_elbow, cls.r_shoulder,
                cls.l_shoulder, cls.l_elbow, cls.l_wrist,
                ]

    @classmethod
    def to_posetriplet_order(cls):
        """
        Matches the posetriplet 16 point order
        :return:
        """
        return [cls.pelvis, cls.r_hip, cls.r_knee, cls.r_ankle,
                cls.l_hip, cls.l_knee, cls.l_ankle,
                cls.torso, cls.neck, cls.head,
                cls.l_shoulder, cls.l_elbow, cls.l_wrist,
                cls.r_shoulder, cls.r_elbow, cls.r_wrist,
                ]

    def __init__(self):
        pass

    @classmethod
    def names(cls):
        return cls._names


aspset_reference_joints = [APSSet17POrder.l_hip, APSSet17POrder.r_shoulder]

class PosetripletOrder:

    pelvis = 0
    r_hip = 1
    r_knee = 2
    r_ankle = 3
    l_hip = 4
    l_knee = 5
    l_ankle = 6
    torso = 7
    neck = 8
    head = 9
    l_shoulder = 10
    l_elbow = 11
    l_wrist = 12
    r_shoulder = 13
    r_elbow = 14
    r_wrist = 15

    @classmethod
    def to_aspset_order(cls):
        return [cls.r_ankle, cls.r_knee, cls.r_hip,
                cls.r_wrist, cls.r_elbow, cls.r_shoulder,
                cls.l_ankle, cls.l_knee, cls.l_hip,
                cls.l_wrist, cls.l_elbow, cls.l_shoulder,
                cls.head, cls.head, cls.neck, cls.torso, cls.pelvis]

    @classmethod
    def to_our_17p_order(cls):
        """
        Matches our 17 point order
        :return:
        """
        return [cls.r_ankle, cls.r_knee, cls.r_hip,
                cls.l_hip, cls.l_knee, cls.l_ankle,
                cls.pelvis,
                cls.neck, cls.torso,
                cls.head, cls.head,
                cls.r_wrist, cls.r_elbow, cls.r_shoulder,
                cls.l_shoulder, cls.l_elbow, cls.l_wrist,
                ]

    _bodypart_indices = [[head, neck],
                         [neck, torso], [torso, pelvis],
                         [neck, r_shoulder], [r_shoulder, r_elbow], [r_elbow, r_wrist],
                         [neck, l_shoulder], [l_shoulder, l_elbow], [l_elbow, l_wrist],
                         [pelvis, r_hip], [r_hip, r_knee], [r_knee, r_ankle],
                         [pelvis, l_hip], [l_hip, l_knee], [l_knee, l_ankle],
                         ]

class MSCOCOJointOrder:
    """
    Helper for the MS COCO 17-point pose definition.
    """

    nose = 0
    l_eye = 1
    r_eye = 2
    l_ear = 3
    r_ear = 4
    l_shoulder = 5
    r_shoulder = 6
    l_elbow = 7
    r_elbow = 8
    l_wrist = 9
    r_wrist = 10
    l_hip = 11
    r_hip = 12
    l_knee = 13
    r_knee = 14
    l_ankle = 15
    r_ankle = 16

    num_points = 17
    num_bodyparts = 12

    @classmethod
    def to_aspset_17p_order(cls):
        """
        Matches the ASPSet 17point order
        :return:
        """
        return [cls.r_ankle, cls.r_knee, cls.r_hip,
                cls.r_wrist, cls.r_elbow, cls.r_shoulder,
                cls.l_ankle, cls.l_knee, cls.l_hip,
                cls.l_wrist, cls.l_elbow, cls.l_shoulder,
                ]

def convert_AlphaOpenposeCoco_to_standard16Joint(pose):
    """
    pose_x: nx17x2
    https://zhuanlan.zhihu.com/p/367707179
    """
    hip = 0.5 * (pose[:, 11] + pose[:, 12])
    neck = 0.5 * (pose[:, 5] + pose[:, 6])
    spine = 0.5 * (neck + hip)

    # head = 0.5 * (pose_x[:, 1] + pose_x[:, 2])

    head_0 = pose[:, 0]  # by noise
    head_1 = (neck - hip)*0.5 + neck  # by backbone
    head_2 = 0.5 * (pose[:, 1] + pose[:, 2])  # by two eye
    head_3 = 0.5 * (pose[:, 3] + pose[:, 4])  # by two ear
    head = head_0 * 0.1 + head_1 * 0.6 + head_2 * 0.1 + head_3 * 0.2

    combine = np.stack([hip, spine, neck, head])  # 0 1 2 3 ---> 17, 18, 19 ,20
    combine = np.transpose(combine, (1, 0, 2))
    combine = np.concatenate([pose, combine], axis=1)
    reorder = [17, 12, 14, 16, 11, 13, 15, 18, 19, 20, 5, 7, 9, 6, 8, 10]
    standart_16joint = combine[:, reorder]
    return standart_16joint