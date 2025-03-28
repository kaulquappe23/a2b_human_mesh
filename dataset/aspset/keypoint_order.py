# -*- coding: utf-8 -*-
"""
Created on 15 Jun 2022, 10:53

"""

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

    def __init__(self):
        pass

    @classmethod
    def names(cls):
        return cls._names


aspset_reference_joints = [APSSet17POrder.l_hip, APSSet17POrder.r_shoulder]
