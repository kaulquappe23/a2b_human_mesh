# -*- coding: utf-8 -*-
"""
Created on 26.02.24

@author: Katja

"""
import csv

import numpy as np

from dataset.aspset.keypoint_order import APSSet17POrder
from dataset.fit3d.keypoint_order_fit3d import Fit3DOrder


class SkipCSVCommentsIterator:
    """
    Simple file-iterator wrapper to skip empty and '#'-prefixed lines.
    Taken from https://bytes.com/topic/python/answers/513222-csv-comments
    (User: skip)
    """

    def __init__(self, fp):
        self.fp = fp

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.fp)
        if not line.strip() or line[0] == "#":
            return next(self)
        return line

def create_dictionary(csv_return):
    result_dict = {}
    for data in zip(*csv_return):
        key = ""
        for i in range(len(data) - 1):
            key = key + str(data[i]) + ";"
        result_dict[key[:-1]] = data[-1]
    return result_dict


def equalize_order(keep_order_csv_return, reorder_csv_return):
    reorder_dict = create_dictionary(reorder_csv_return)
    result = []
    for data in zip(*keep_order_csv_return):
        key = ""
        for i in range(len(data) - 1):
            key = key + str(data[i]) + ";"
        result.append(reorder_dict[key[:-1]])
    return np.asarray(result)

def read_aspset_csv_annotations(csv_path, num_joints=APSSet17POrder.num_points):
    """
    Reads annotations from the given csv file. Additional info to be expected is dataset;image_path;video_description;frame_num
    The first line has to be a comment with the number of annotations, eg. "#123"

    Parameters:
        - csv_path: Path to the csv file

    returns_
        - List of subject names
        - List of clip names
        - List of camera names
        - List of frame numbers
        - Numpy.ndarray (num_images x 17 x 3) with the joint locations and the "visibility" flag.
    """
    with open(csv_path, "r") as f:
        first_line = f.readline()
        num_annotations = int(first_line.strip("#").strip("\n").strip("\r").strip(";"))

    offset = 4
    subjects = list()
    clips = list()
    cameras = list()
    frame_nums = list()
    joints = np.ndarray(shape=(num_annotations, num_joints, 3), dtype=float)

    with open(csv_path, "r") as f:
        reader = csv.reader(SkipCSVCommentsIterator(f), delimiter=';')
        row_count = 0
        skip_header = True
        for row in reader:
            if skip_header:
                skip_header = False
                continue
            assert (len(row) == offset + 3 * num_joints)
            subjects.append(row[0])
            clips.append(row[1])
            cameras.append(row[2])
            frame_nums.append(int(row[3]))
            joint_count = 0
            for i in range(offset + 2, len(row), 3):
                joints[row_count, joint_count] = [float(row[i - 2]), float(row[i - 1]), float(row[i])]
                joint_count += 1
            row_count += 1

    return subjects, clips, cameras, frame_nums, joints

def read_fit3d_csv_annotations(csv_path, eval_all_joints=False):
    """
    Reads annotations from the given csv file. Additional info to be expected is dataset;image_path;video_description;frame_num
    The first line has to be a comment with the number of annotations, eg. "#123"

    Parameters:
        - csv_path: Path to the csv file

    returns_
        - List of subject names
        - List of clip names
        - List of camera names
        - List of frame numbers
        - Numpy.ndarray (num_images x 17 x 3) with the joint locations and the "visibility" flag.
    """
    with open(csv_path, "r") as f:
        first_line = f.readline()
        num_annotations = int(first_line.strip("#").strip("\n").strip("\r").strip(";"))

    offset = 4
    subjects = list()
    actions = list()
    cameras = list()
    frame_nums = list()
    joints = np.ndarray(shape=(num_annotations, 144, 3), dtype=float)

    with open(csv_path, "r") as f:
        reader = csv.reader(SkipCSVCommentsIterator(f), delimiter=';')
        row_count = 0
        skip_header = True
        for row in reader:
            if skip_header:
                skip_header = False
                continue
            assert (len(row) == offset + 3 * 144)
            subjects.append(row[0])
            cameras.append(row[1])
            actions.append(row[2])
            frame_nums.append(int(row[3]))
            joint_count = 0
            for i in range(offset + 2, len(row), 3):
                joints[row_count, joint_count] = [float(row[i - 2]), float(row[i - 1]), float(row[i])]
                joint_count += 1
            row_count += 1

    if not eval_all_joints:
        joints = joints[:, Fit3DOrder.from_SMPLX_order()]

    return subjects, cameras, actions, frame_nums, joints