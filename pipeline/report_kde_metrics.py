import sys
import random
import string
from pathlib import Path
import numpy as np
import pandas as pd
import glob
import natsort
from scipy import spatial

import torch
import trimesh

# from pipeline.p_utils.pipeline_utils import parse_config, seed_everything, get_train_val_sets

sys.path.append("./Partial Registration/")
sys.path.append("./Partial Registration/SdfVoxelMatching/")
sys.path.append("./Partial Registration/SdfVoxelMatching/pytorch-3dunet/")
# import encoder

from slice_matching_meanshape import run_iterative_slice_matching_experiment_meanshape


def array_eval(x):
    # horrible security vulnerability, but what can I do...
    nan = float("nan")
    array = np.array
    return eval(x)

def read_csv(path):
    return pd.read_csv(path, converters={"candidate_slice_params": array_eval})

mm_factor = 0.12

def write_results(input_path, output_path_txt, output_path_csv, diameter_levels_by_percent = {10: 33.74}):
    results_df = read_csv(input_path)
    results_best_candidate = results_df.loc[results_df["candidate_index"] == 0]
    results = [
        results_best_candidate["centers_dist"].mean(),
        results_best_candidate["centers_dist"].std(),
        results_best_candidate["angle"].mean(),
        results_best_candidate["angle"].std(),
        results_best_candidate["candidate_slices_mean_distance"].mean(),
        results_best_candidate["candidate_slices_mean_distance"].std(),
    ] + [
        (results_best_candidate["centers_dist"] <= diameter_level).mean() * 100
        for diameter_level in diameter_levels_by_percent.values()
    ]

    with open(output_path_txt, "w") as text_file:
        text_file.writelines([
            f"center_dist mean:                                                 {results[0]:8.4f}  +/- {results[1]:8.4f}  | {(results[0]*mm_factor):6.4f}mm +/- {(results[1]*mm_factor):6.4f}mm\n",
            f"angle mean:                                                       {results[2]:8.4f}° +/- {results[3]:8.4f}°\n",
            f"candidate_slice_mean_distance:                                    {results[4]:8.4f}  +/- {results[5]:8.4f}  | {(results[4]*mm_factor):6.4f}mm +/- {(results[5]*mm_factor):6.4f}mm\n",
        ] + [
            f"percentage with distance error <= {percent}% diameter ({diameter_level:5.2f} | {(diameter_level*mm_factor):6.3f}mm): {diameter_result}%\n"
            for (percent, diameter_level), diameter_result in zip(diameter_levels_by_percent.items(), results[6:])
        ])

    pd.DataFrame({"results": results}).to_csv(output_path_csv)
    return results

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage: python report_kde_metrics.py EXPERIMENT_PATH1 ... EXPERIMENT_PATHN")
        print("Experiment paths are the paths one level above the fold folders")
    # A list of paths to experiment folders (one level above the fold_x folder) should be parsed as arguments
    for experiment_path in sys.argv[1:]:
        fold_paths = natsort.natsorted(glob.glob(f"{experiment_path}/sdf_train_fold_*"))
        input_csv_files = natsort.natsorted(glob.glob(f"{experiment_path}/sdf_train_fold_*/output_matching_sliceMatchingExperiment.csv"))
        if len(fold_paths) != 4:
            print(f"Skipping {experiment_path}: {'less' if len(fold_paths) < 4 else 'more'} than 4 folds")
            continue
        if len(input_csv_files) != 4:
            print(f"Skipping {experiment_path}: not exactly 4 csv files")
            continue
        fold_results = []
        for fold_path, input_csv_file in zip(fold_paths, input_csv_files):
            meanshape = trimesh.load_mesh(Path(fold_path) / "meansdf_val" / "meanshape.ply")
            diameter = spatial.distance_matrix(meanshape.vertices, meanshape.vertices).max()

            diameter_levels_by_percent = {10: diameter * 0.1, 15: diameter * 0.15, 20: diameter * 0.2}
            fold_results.append(write_results(input_csv_file, Path(fold_path) / "sliceMatchingMetrics.txt", Path(fold_path) / "sliceMatchingMetrics.csv", 
                diameter_levels_by_percent))
        fold_results = np.array(fold_results) # rows are different experiments
        fold_avg = np.mean(fold_results, axis=0)[[0,2,4]+list(range(6,6+len(diameter_levels_by_percent)))]
        fold_std = np.std(fold_results, axis=0)[[0,2,4]+list(range(6,6+len(diameter_levels_by_percent)))]
        with open(Path(experiment_path) / "sliceMatchingMetricsFolds.txt", "w") as text_file:
            text_file.writelines([
                f"center_dist mean:                              {fold_avg[0]:8.4f}  +/- {fold_std[0]:8.4f}  | {(fold_avg[0]*mm_factor):6.4f}mm +/- {(fold_std[0]*mm_factor):6.4f}mm\n",
                f"angle mean:                                    {fold_avg[1]:8.4f}° +/- {fold_std[1]:8.4f}°\n",
                f"candidate_slice_mean_distance:                 {fold_avg[2]:8.4f}  +/- {fold_std[2]:8.4f}  | {(fold_avg[2]*mm_factor):6.4f}mm +/- {(fold_std[2]*mm_factor):6.4f}mm\n",
            ] + [
                f"percentage with distance error <= {percent}% diameter: {diameter_avg:5.2f}% +/- {diameter_std:5.2f}%\n"
                for percent, diameter_avg, diameter_std in zip(diameter_levels_by_percent.keys(), fold_avg[3:], fold_std[3:])
            ])
        pd.DataFrame({"mean": fold_avg, "std.dev.": fold_std}).to_csv(Path(experiment_path) / "sliceMatchingMetricsFolds.csv")
        print(f"Reported results for {experiment_path}")
