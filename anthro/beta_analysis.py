# -*- coding: utf-8 -*-
"""
Created on 09.08.24

@author: Katja

"""
import os
import pickle
from collections import defaultdict

import numpy as np
import smplx
import torch
from tqdm import tqdm

from config import SMPL_MODEL_DIR


def analyze_betas(beta_params, calc_lengths=True, model_gender="neutral", save_path=None, force_recalc=True):
    """
    Analyze beta parameters
    :param save_path: Path to save anthropometric measurements
    :param model_gender: gender of model to use for beta params
    :param calc_lengths: calculate anthropometric measurements if set to true, otherwise only median values are generated
    :param beta_params: dict mapping subject -> beta param list or path with pickled dict
    :param force_recalc: force recalculation of anthropometric measurements, otherwise the saved results are loaded
    :return:
    """

    if isinstance(beta_params, str):
        beta_path = beta_params
        with open(beta_path, "rb") as f:
            beta_params = pickle.load(f)
            if save_path is None:
                save_path = beta_path[:-4] + "_measures"

    np.set_printoptions(formatter={'float': '{: .3f}'.format})

    if save_path is not None:
        if save_path.endswith(".pkl"):
            save_path = save_path[:-4]
    file_exists = os.path.exists(f"{save_path}.pkl")
    if file_exists:
        with open(f"{save_path}.pkl", "rb") as f:
            final_dict = pickle.load(f)
    else:
        final_dict = {}
    for s in beta_params:
        if isinstance(beta_params[s][0], torch.Tensor):
            beta_params[s] = [t.cpu().numpy() for t in beta_params[s]]
        betas = np.asarray(beta_params[s])
        if len(betas.shape) == 3:
            betas = betas[:, 0]
        print(f"Subject {s} - {betas.shape[0]} examples:")
        print(f"  Mean beta: {np.average(betas, axis=0)}")
        print(f"Median beta: {np.median(betas, axis=0)}")
        print(f"   Max beta: {np.max(betas, axis=0)}")
        print(f"   Min beta: {np.min(betas, axis=0)}")
        print(f"Stddev beta: {np.std(betas, axis=0)}")
        print(f"Mean stddev beta: {np.mean(np.std(betas, axis=0)):.2f}")

        if calc_lengths and force_recalc and not file_exists:

            model = smplx.create(SMPL_MODEL_DIR, model_type="smplx",
                                         gender=model_gender, use_face_contour=False,
                                         num_betas=10)
            print()
            betas = np.asarray(beta_params[s])
            measures = []
            for beta_id in tqdm(range(betas.shape[0])):
                measure_res = smplx_measurer.measure(torch.from_numpy(betas[beta_id])[None].float(), model=model)
                measures.append(measure_res)

            measures = np.asarray(measures)[:, 0]
            measure_dict = smplx_measurer.measurement_array_to_dict(measures)

            for measure_key, measure_vals in measure_dict.items():
                # IN CM!!
                print(f"{measure_key:>30}: avg {np.average(measure_vals)*100:.3f}, median {np.median(measure_vals)*100:.3f}, max {np.max(measure_vals)*100:.3f}, min {np.min(measure_vals)*100:.3f}, stddev {np.std(measure_vals*100):.3f}")
            measure_dict["original_median_betas"] = np.median(betas, axis=0)
            print()
            print()
            final_dict[s] = measure_dict
        elif calc_lengths:
            overall = {"stddev": defaultdict(list), "stddev_rel": defaultdict(list), "rel_range": defaultdict(list)}
            for measure_key, measure_vals in final_dict[s].items():
                if measure_key == "original_median_betas":
                    continue
                # IN CM!!
                print(f"{measure_key:>30}: avg {np.average(measure_vals) * 100:.2f}, median {np.median(measure_vals) * 100:.2f}, "
                      f"max {np.max(measure_vals) * 100:.2f}, min {np.min(measure_vals) * 100:.2f}, "
                      f"stddev {np.std(measure_vals * 100):.2f}, rel. stddev {np.std(measure_vals) / np.average(measure_vals) * 100:.2f}%, "
                      f"rel. range {(np.max(measure_vals) - np.min(measure_vals)) / np.average(measure_vals) * 100:.2f}%")
                overall["stddev"][measure_key].append(np.std(measure_vals * 100))
                overall["stddev_rel"][measure_key].append(np.std(measure_vals) / np.average(measure_vals) * 100)
                overall["rel_range"][measure_key].append((np.max(measure_vals) - np.min(measure_vals)) / np.average(measure_vals) * 100)
            overall["stddev"]["betas"].extend(np.std(betas, axis=0))
    if file_exists and calc_lengths:
        print(f"\n\nOverall:")
        for measure_key in overall["stddev_rel"]:
            print(f"{measure_key:>30} - Stddev  : {np.average(overall['stddev'][measure_key]):.2f}, Stddev r: {np.average(overall['stddev_rel'][measure_key]):.2f}, R. range: {np.average(overall['rel_range'][measure_key]):.2f}")
        print(f"Stddev beta : {np.average(overall['stddev']['betas']):.2f}")

    if save_path is not None:
        with open(f"{save_path}.pkl", "wb") as f:
            pickle.dump(final_dict, f)