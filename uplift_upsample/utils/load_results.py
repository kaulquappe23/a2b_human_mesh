# -*- coding: utf-8 -*-
"""
Created on 09.08.24

@author: Katja

"""
import pickle

import numpy as np


def load_uu_results(path):
    with open(path, "rb") as f:
        pred_keypoints3d, subjects, actions, indices, cam_params, gt_keypoints3d, conversion_dicts, cam_names = pickle.load(f)

        subj_dict = {}
        action_dict = {}
        for s, s_id in conversion_dicts[0].items():
            subj_dict[s_id] = s
        for c, c_id in conversion_dicts[1].items():
            action_dict[c_id] = c

        pred_dict = {}
        for pred_kp, s, a, c, frame in zip(pred_keypoints3d, subjects, actions, cam_names, indices):
            s = subj_dict[s]
            a = action_dict[a]
            if s not in pred_dict:
                pred_dict[s] = {}
            if c not in pred_dict[s]:
                pred_dict[s][c] = {}
            if a not in pred_dict[s][c]:
                pred_dict[s][c][a] = []
            assert len(pred_dict[s][c][a]) == frame
            pred_dict[s][c][a].append(pred_kp)

        for s in pred_dict:
            for c in pred_dict[s]:
                for a in pred_dict[s][c]:
                    pred_dict[s][c][a] = np.stack(pred_dict[s][c][a])
    return pred_dict