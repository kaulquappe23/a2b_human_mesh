# -*- coding: utf-8 -*-
"""
Created on 19.06.24

@author: Katja

"""
import pickle

import numpy as np
import torch


def get_all_beta_params(filename):
    """
    Read anthropometric measurements from file and return as torch tensor
    :param filename: file to read
    :return:
    """
    with open(filename, "rb") as f:
        all_params = pickle.load(f)
    beta_params = {}
    for beta_param_name in all_params:
        beta_params[beta_param_name] = all_params[beta_param_name]
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for k in beta_params[beta_param_name]:
            if k not in ["name", "gender", "num_betas"]:
                beta_params[beta_param_name][k] = beta_params[beta_param_name][k]
                if not isinstance(beta_params[beta_param_name][k], torch.Tensor):
                    beta_params[beta_param_name][k] = torch.from_numpy(np.asarray(beta_params[beta_param_name][k])).to(device).float()
                else:
                    beta_params[beta_param_name][k] = beta_params[beta_param_name][k].to(device).float()
    return beta_params
