import sys
import random
import string
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import trimesh

from pipeline.p_utils.pipeline_utils import parse_config, seed_everything, get_train_val_sets

sys.path.append("./Partial Registration/")
sys.path.append("./Partial Registration/SdfVoxelMatching/")
sys.path.append("./Partial Registration/SdfVoxelMatching/pytorch-3dunet/")
import encoder

from slice_matching_meanshape import run_iterative_slice_matching_experiment_meanshape


def array_eval(x):
    # horrible security vulnerability, but what can I do...
    nan = float("nan")
    array = np.array
    return eval(x)

def read_csv(path):
    return pd.read_csv(path, converters={"candidate_slice_params": array_eval})

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
            f"center_dist mean:                                      {results[0]} +/- {results[1]}",
            f"angle mean:                                            {results[2]} +/- {results[3]}",
            f"candidate_slice_mean_distance:                         {results[4]} +/- {results[5]}",
        ] + [
            f"percentage with distance error <= {percent} percent diameter: {diameter_result}"
            for percent, diameter_result in zip(diameter_levels_by_percent.keys(), results[6:])
        ])

    pd.DataFrame({"results": results}).to_csv(output_path_csv)
    return results


def run_kde_eval(config, data_dir, output_path, model_path, val_idx, device="cuda"):
    model = torch.load(model_path, map_location=torch.device(device))
    if isinstance(model, torch.nn.parallel.DataParallel):
        model = model.module

    ### Original dataset
    us_data_paths = [f"{data_dir}/{us_index}.nii" for us_index in val_idx]
    mesh_paths = [f"{data_dir}/{us_index}.ply" for us_index in val_idx]

    print(us_data_paths)
    print(mesh_paths)

    mesh_path = experiment_path / "meansdf_val" / "meanshape.ply"
    assert mesh_path.exists(), f"{mesh_path} does not exist"
    meanshape_mesh = trimesh.exchange.load.load(mesh_path)
    sdf_path = experiment_path / "meansdf_val" / "SSMsample_0_sdf"
    assert sdf_path.exists(), f"{sdf_path} does not exist"
    meanshape_sdf = torch.load(sdf_path)

    # ##### weighted soft margine triplet loss
    run_iterative_slice_matching_experiment_meanshape(
        model=model, num_sdf_patches=config.num_sdf_patches, num_slice_patches=config.num_slice_patches,
        model_path=model_path, us_data_paths=us_data_paths, mesh_paths=mesh_paths, mean_shape_sdf=meanshape_sdf,
        mean_shape_mesh=meanshape_mesh, device=device, smallest_k=config.smallest_k,
        slice_patch_size=np.array(config.US_patch_shape), sdf_patch_size=np.array(config.sdf_patch_shape),
        z_restriction_width=config.z_restriction_width, kde_bandwidth=config.kde_bandwidth,
        kde_k_largest=5, verbose=False, sdf_batch_size=config.sdf_batch_size,
        output_path=experiment_path, hungarian_matching=config.hungarian_matching,
        log_filename="sliceMatchingExperiment.csv")

    write_results(experiment_path / "output_matching_sliceMatchingExperiment.csv",
                  experiment_path / "sliceMatchingMetrics.txt",
                  experiment_path / "sliceMatchingMetrics.csv")

if __name__ == "__main__":
    config = parse_config()
    seed_everything(config.SEED)

    # get folds
    val_fold = config.data.val_fold
    eval_path = config.kde.eval_path
    assert f"fold_{val_fold}" in eval_path, f"fold_{val_fold} not in {eval_path}"
    train_indices, val_indices = get_train_val_sets(config, val_fold)

    # find the best model
    experiment_path = Path(config.output_path) / config.reg.path_prefix / eval_path
    save_path =  experiment_path / 'saves'
    best_model = list(save_path.glob("best*.obj"))
    assert len(best_model) == 1, f"errant best models found {best_model}"

    run_kde_eval(config.kde, Path(config.data_path), experiment_path, str(best_model[0]), val_indices)
