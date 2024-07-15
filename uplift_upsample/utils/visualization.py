# -*- coding: utf-8 -*-
"""
Created on 12.12.23

@author: Katja

"""
import colorsys
from io import BytesIO

import cv2
import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt

from dataset.h36m.keypoint_order import H36MOrder17P


def create_body_part_mesh(x, y, z, radius=0.02):
    return pv.Sphere(radius=radius, center=(x, y, z))


def create_skeleton_line(x1, y1, z1, x2, y2, z2):
    return pv.Line((x1, y1, z1), (x2, y2, z2))


def pyvista_3d_pose(joints_3d, bodyparts, size=350):
    plotter = pv.Plotter(notebook=False, off_screen=True, window_size=[size, size])

    for x, y, z in joints_3d:
        mesh = create_body_part_mesh(x, y, z)
        plotter.add_mesh(mesh)

    for i, j in bodyparts:
        x1, y1, z1 = joints_3d[i]
        x2, y2, z2 = joints_3d[j]
        line = create_skeleton_line(x1, y1, z1, x2, y2, z2)
        plotter.add_mesh(line, line_width=0.01)

    plotter.camera.focal_point = [0, 0, 0]
    plotter.background_color = '#dddddd'
    plotter.view_isometric()
    return plotter


def matplotlib_3d_pose(pose, bodyparts, colors, az=0, elev=20, dist=9, ax_lim=0.8):
    pose = np.copy(pose)
    pose -= pose[H36MOrder17P.pelvis]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.azim = az
    ax.elev = elev
    ax.dist = dist

    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], color='0.2')

    for bodypart, color in zip(bodyparts, colors):
        start_joint, end_joint = pose[bodypart[0]], pose[bodypart[1]]
        x_vals, y_vals, zvals = list(zip(start_joint, end_joint))
        color = [c / 255. for c in color]
        plt.plot(x_vals, y_vals, zvals, color=tuple(color), linewidth=4.)

    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return fig


def matplotlib_2d_pose(p2d, bodyparts, colors):

    p2d = np.copy(p2d)
    fig = plt.figure()

    scale = 0.5
    min_x, min_y = np.round(np.min(p2d, axis=0)).astype(np.int32)
    max_x, max_y = np.round(np.max(p2d, axis=0)).astype(np.int32)
    w = max_x - min_x
    h = max_y - min_y
    w_widen = int(.3 * w)
    h_widen = int(.25 * h)
    crop_box = [min_x - w_widen, max_x + w_widen,
                min_y - h_widen, max_y + h_widen]
    min_x, max_x, min_y, max_y = crop_box
    crop = np.ones((max_y - min_y, max_x - min_x, 3)).astype(np.float32) * 255.
    crop = crop.astype(np.uint8)

    p2d[:, 0] -= min_x
    p2d[:, 1] -= min_y

    p2d *= scale
    crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))

    plt.imshow(crop)

    for joint in p2d:
        color = 'k'
        plt.scatter(joint[0], joint[1], s=5, color=color, zorder=2)

    for bodypart, color in zip(bodyparts, colors):
        start_joint, end_joint = p2d[bodypart[0]], p2d[bodypart[1]]
        x_vals, y_vals = list(zip(start_joint, end_joint))
        color = [c / 255. for c in color]
        plt.plot(x_vals, y_vals, color=tuple(color), linewidth=2, zorder=1)

    # plt.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png")
    return buf


def _hsv_colors(num_colors, hue_range=[0., 360.], lightness=60, saturation=95):
    """
    Taken from: http://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
    (Uri Cohen)
    """
    colors = []
    lightness = lightness / 100.0
    saturation = saturation / 100.0
    for i in np.arange(hue_range[0], hue_range[1], (hue_range[1] - hue_range[0]) / num_colors):
        hue = i / 360.
        color = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(tuple([int(c * 255) for c in color]))
    return colors


class Pose17PointBodypartColors:
    __num_body_part_colors = 6
    __raw_hsv_colors = _hsv_colors(__num_body_part_colors, saturation=80)

    head = __raw_hsv_colors[0]
    neck = __raw_hsv_colors[0]
    upper_torso = __raw_hsv_colors[4]
    lower_torso = __raw_hsv_colors[4]
    r_shoulder = __raw_hsv_colors[4]
    r_upper_arm = __raw_hsv_colors[1]
    r_lower_arm = __raw_hsv_colors[3]
    l_shoulder = __raw_hsv_colors[4]
    l_upper_arm = __raw_hsv_colors[1]
    l_lower_arm = __raw_hsv_colors[3]
    r_hip = __raw_hsv_colors[4]
    r_upper_leg = __raw_hsv_colors[2]
    r_lower_leg = __raw_hsv_colors[5]
    l_hip = __raw_hsv_colors[4]
    l_upper_leg = __raw_hsv_colors[2]
    l_lower_leg = __raw_hsv_colors[5]

    colors = [head, neck,
              upper_torso, lower_torso,
              r_shoulder, r_upper_arm, r_lower_arm,
              l_shoulder, l_upper_arm, l_lower_arm,
              r_hip, r_upper_leg, r_lower_leg,
              l_hip, l_upper_leg, l_lower_leg
              ]

    stroked = [False, False,
               False, False,
               False, False, False,
               True, True, True,
               False, False, False,
               True, True, True,
               ]

    def __init__(self):
        pass
