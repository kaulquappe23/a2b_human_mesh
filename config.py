# -*- coding: utf-8 -*-
"""
Created on 22.05.24

@author: Katja

"""
import os.path

SMPL_MODEL_DIR = "./smpl-models"
assert os.path.exists(SMPL_MODEL_DIR), f"SMPL model directory {SMPL_MODEL_DIR} does not exist"
VPOSER_DIR = './inverse_kinematics/V02_05'
ASPSET_REGRESSOR_PATH = "./dataset/aspset/regressor/aspset_regressor_v5.npy"
VERT_SEGM_PATH = "./anthro/measurements/smplx_vert_segmentation.json"
CIRCUMFERENCE_CONFIG_PATH = "./anthro/measurements/SMPLX-circumference.config"