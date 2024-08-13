"""
This module contains different ways of measuring lengths between
groups of points
"""

# external libraries
import numpy as np

from anthro.measurements.util import quick_index, find_most_equidistant, find_max_len


class Length(object):
    """
    computes the lengths between points as a strategy class,
    is a baseclass for other length measurements,

    works in batches
    """
    def __init__(self, from_, to):
        self.from_ = from_
        self.to = to
        self.compd = False # flag, whether this object is "compiled"
        self.from_c = None
        self.to_c = None

    def comp(self, indices):
        self.compd = True
        self.from_c = quick_index(indices, self.from_)
        self.to_c = quick_index(indices, self.to)


    @staticmethod
    def add_direction(indices, config, dir = "l"):
        if type(config) is str:
            r = dir + config
            if r in indices or len(indices) == 0:
                return r
            return config
        if type(config) is tuple or type(config) is list:
            return [Length.add_direction(indices, i, dir) for i in config]
        return config


    def get_dir(self, indices):
        """
        gives back the directional versions of the length measure
        """
        return (type(self)(Length.add_direction(indices, self.from_, "l"),
                           Length.add_direction(indices, self.to,    "l")),
                type(self)(Length.add_direction(indices, self.from_, "r"),
                           Length.add_direction(indices, self.to,    "r")))


    def calc(self, vertices: np.ndarray, visualize = False):
        a = vertices[..., self.from_c,:].mean(axis=-2)
        b = vertices[..., self.to_c  ,:].mean(axis=-2)
        res = np.linalg.norm(a - b, axis = -1)
        if visualize:
            return res, (a[0], b[0])
        return res


class Height(Length):
    """
    computes the height difference between points as a strategy class,
    works in batches
    """
    def __init__(self, from_, to):
        super().__init__(from_, to)

    def calc(self, vertices: np.ndarray, visualize = False):
        a = vertices[..., self.from_c,:].mean(axis=-2)
        b = vertices[..., self.to_c  ,:].mean(axis=-2)
        res = a[...,1] - b[...,1]
        if visualize:
            b[..., 0] = a[..., 0]
            b[..., 2] = a[..., 2]
            return res, (a[0], b[0])
        return res

class Middle(Length):
    """
    computes the difference between the most equidistant points of each
    point cloud as a strategy class,
    works in batches
    """
    def __init__(self, from_, to):
        super().__init__(from_, to)

    def calc(self, vertices: np.ndarray, visualize = False):
        a, _ = find_most_equidistant(vertices[..., self.from_c,:])
        b, _ = find_most_equidistant(vertices[..., self.to_c  ,:])
        res = np.linalg.norm(a - b, axis = -1)
        if visualize:
            return res, (a[0], b[0])
        return res

class Furthest(Length):
    """
    computes the difference between the furthest points of each
    point cloud as a strategy class,
    works in batches
    """
    def __init__(self, from_, to):
        super().__init__(from_, to)

    def calc(self, vertices: np.ndarray, visualize = False):
        a, _ = vertices[..., self.from_c,:]
        b, _ = vertices[..., self.to_c  ,:]
        res = find_max_len(a, b)
        if visualize:
            return res, (a[0], b[0])
        return res
