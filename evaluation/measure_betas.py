# -*- coding: utf-8 -*-
"""
Created on 13.08.24

@author: Katja

"""
import os
import pickle
from collections import defaultdict

import numpy as np
import smplx
import torch
from tqdm import tqdm

from anthro.measurements.measure import smplx_measurer
from config import SMPL_MODEL_DIR


def analyze_betas(beta_params, model_gender="neutral", save_path=None, force_recalc=True):
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

    overall = {"stddev": defaultdict(list), "stddev_rel": defaultdict(list), "rel_range": defaultdict(list)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for s in beta_params:
        if isinstance(beta_params[s][0], torch.Tensor):
            beta_params[s] = [t.cpu().numpy() for t in beta_params[s]]
        betas = np.asarray(beta_params[s])
        if len(betas.shape) == 3:
            betas = betas[:, 0]
        print(f"\nSubject {s} - {betas.shape[0]} examples:")
        print(f"  Mean beta: {np.average(betas, axis=0)}")
        print(f"Median beta: {np.median(betas, axis=0)}")
        print(f"   Max beta: {np.max(betas, axis=0)}")
        print(f"   Min beta: {np.min(betas, axis=0)}")
        print(f"Stddev beta: {np.std(betas, axis=0)}")
        print(f"Mean stddev beta: {np.mean(np.std(betas, axis=0)):.2f}")

        if force_recalc or not file_exists:

            model = smplx.create(SMPL_MODEL_DIR, model_type="smplx",
                                         gender=model_gender, use_face_contour=False,
                                         num_betas=10).to(device)
            print()
            betas = np.asarray(beta_params[s])
            measures = []
            for beta_id in tqdm(range(betas.shape[0])):
                measure_res = smplx_measurer.measure(torch.from_numpy(betas[beta_id])[None].float().to(device), model=model)
                measures.append(measure_res)

            measures = np.asarray(measures)[:, 0]
            measure_dict = smplx_measurer.measurement_array_to_dict(measures)

            measure_dict["original_median_betas"] = np.median(betas, axis=0)
            print()
            print()
            final_dict[s] = measure_dict

        for measure_key, measure_vals in final_dict[s].items():
            if measure_key == "original_median_betas":
                continue
            # IN CM!!
            print(f"{measure_key:>30}: avg {np.average(measure_vals) * 100:.2f}cm, median {np.median(measure_vals) * 100:.2f}cm, "
                  f"max {np.max(measure_vals) * 100:.2f}cm, min {np.min(measure_vals) * 100:.2f}cm, "
                  f"stddev {np.std(measure_vals * 100):.2f}cm, rel. stddev {np.std(measure_vals) / np.average(measure_vals) * 100:.2f}%, "
                  f"rel. range {(np.max(measure_vals) - np.min(measure_vals)) / np.average(measure_vals) * 100:.2f}%")
            overall["stddev"][measure_key].append(np.std(measure_vals * 100))
            overall["stddev_rel"][measure_key].append(np.std(measure_vals) / np.average(measure_vals) * 100)
            overall["rel_range"][measure_key].append((np.max(measure_vals) - np.min(measure_vals)) / np.average(measure_vals) * 100)
        overall["stddev"]["betas"].extend(np.std(betas, axis=0))

    print(f"\n\n---------------- Overall (Average over all subjects): ----------------")
    for measure_key in overall["stddev_rel"]:
        print(f"{measure_key:>30} - Stddev  : {np.average(overall['stddev'][measure_key]):.2f}, Stddev r: {np.average(overall['stddev_rel'][measure_key]):.2f}, R. range: {np.average(overall['rel_range'][measure_key]):.2f}")
    print(f"Stddev beta : {np.average(overall['stddev']['betas']):.2f}")

    if save_path is not None:
        if not save_path.endswith(".pkl"):
            save_path += ".pkl"
        with open(save_path, "wb") as f:
            pickle.dump(final_dict, f)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='Beta parameter measurement.')
    parser.add_argument('--input', required=True,
                        help="File with beta parameters in correct dict format: mapping of subject name to list of beta parameters")
    parser.add_argument('--output', required=True,
                        help="pkl file to store the measurements.")
    parser.add_argument('--gender', required=False,
                        default="neutral",
                        help="Gender of the SMPL-X mesh created with the given beta parameters.")
    parser.add_argument('--recalc', dest='recalc', action='store_true',
                        help="Force recalculation of measurements, otherwise the joints will be loaded if a results file already exists.")

    args = parser.parse_args()

    analyze_betas(args.input, model_gender=args.gender, save_path=args.output, force_recalc=args.recalc)