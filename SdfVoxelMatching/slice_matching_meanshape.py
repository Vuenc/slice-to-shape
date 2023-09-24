import torch
import torch.utils.data
import trimesh
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from encoder import SdfVoxelMatchNet
from slice_matching import (iterative_slice_matching, transform_slice_params, PatchesDataset, compute_embeddings, create_slice_patches_dataset,
    create_sdf_patches_dataset, procrustes_prediction)
from datetime import datetime
from typing import List, Tuple

import os
import sys
from pathlib import Path
sys.path.append("../")
from load_nii_open3d import load_nii_voxels
from voxelgrid_slices import slice_gridpoints
from soft_assignment_slice_matching import soft_assignment_matrix, hard_assignment
import soft_assignment_slice_matching

def compute_angle_centroid_distance_metrics(candidate_params,slice_params_mean_shape):
    slice_center = slice_params_mean_shape[0]
    default_normal = np.array([0,0,1]) # should be normalized
    candidate_normals = np.cross(candidate_params[1], candidate_params[2])
    candidate_slice_centers = candidate_params[0]
    angles = np.arccos((candidate_normals @ default_normal) / np.linalg.norm(candidate_normals)) / (2*np.pi) * 360
    centroid_dists = np.linalg.norm(slice_center - candidate_slice_centers)

    return centroid_dists,angles


def slices_mean_distance(slice_params1, slice_params2, resolution=100):
    slice_gridpoints1 = slice_gridpoints(*slice_params1, num_steps1=resolution, num_steps2=resolution)
    slice_gridpoints2 = slice_gridpoints(*slice_params2, num_steps1=resolution, num_steps2=resolution)
    return np.linalg.norm(slice_gridpoints1 - slice_gridpoints2, axis=0).mean()

def run_iterative_slice_matching_experiment_meanshape(model, num_sdf_patches, num_slice_patches, model_path,
                                                      us_data_paths: List[str], mesh_paths: List[str], mean_shape_sdf: torch.Tensor, mean_shape_mesh: trimesh.Trimesh,
                                                      device, slice_patch_size = np.array([32,32,32]), sdf_patch_size = np.array([32,32,32]),
        num_slices_per_thyroid = 50, smallest_k = None, thresh=None, kde_k_largest = 5, z_restriction_width = 10, kde_bandwidth = 0.05, use_previous_effective_threshold=False, 
        sdf_batch_size=75, iterations=2, verbose=False, output_path=Path("./"), hungarian_matching=False, log_filename=None):
    results_list = []
    torch.cuda.empty_cache()

    # hyperparams = [
    #     ("numSdfPatches", num_sdf_patches),
    #     ("numSlicePatches", num_slice_patches),
    #     ("slicePatch", "x".join(map(str, slice_patch_size))),
    #     ("sdfPatch", "x".join(map(str, sdf_patch_size))),
    #     ("numSlicesPerThyroid", num_slices_per_thyroid),
    #     ("smallestK", smallest_k),
    #     ("thresh", thresh),
    #     ("kdeKLargest", kde_k_largest),
    #     ("zRestrictionWidth", z_restriction_width),
    #     ("kdeBandwidth", kde_bandwidth),
    #     ("model", model_path[model_path.rfind("/")+1+6:model_path.rfind("/")+1+20]),
    #     (None, "prevEffThresh" if use_previous_effective_threshold else "noPrevEffThresh"),
    #     ("iter", iterations),
    # ]

    date_str = datetime.now().strftime("%b%d_%H-%M-%S")
    if log_filename is None:
        log_filename = "sliceMatchingExperiment_" + date_str + ".csv"
    print(len(log_filename))
    print(log_filename)

    for data_index, (us_path, mesh_path) in tqdm(enumerate(zip(us_data_paths, mesh_paths))):
        ultrasound = load_nii_voxels(us_path)
        us_mesh: trimesh.Trimesh  = trimesh.load_mesh(mesh_path) # type: ignore


        # Define z range for slice samples: not too close to border (z index +- 20 on us, and relatively scaled +-20 on sdf)
        from_us, to_us = us_mesh.vertices[:,2].min(), us_mesh.vertices[:,2].max()
        from_sdf, to_sdf = mean_shape_mesh.vertices[:,2].min(), mean_shape_mesh.vertices[:,2].max()
        scale_factor = (to_sdf - from_sdf) / (to_us - from_us)
        from_us, to_us = from_us + 20 * scale_factor, to_us - 20 * scale_factor
        from_sdf, to_sdf = from_sdf + 20 * scale_factor, to_sdf - 20 * scale_factor
        
        #z_space = np.clip(np.random.randn(num_slices_per_thyroid)*(1/6)+0.5,0,1)
        z_space=np.linspace(0, 1, num_slices_per_thyroid)
        for current_z_index, current_z_factor in tqdm(enumerate(z_space)):
            current_z_us = from_us + current_z_factor * (to_us - from_us)
            current_z_sdf = from_sdf + current_z_factor * (to_sdf - from_sdf)

            slice_params_us = (np.array((us_mesh.vertices[:,0].mean(), us_mesh.vertices[:,1].mean(), current_z_us)), np.array((1.,0,0)), np.array((0,1,0)), 200., 200.)
            # TODO this is simplistic ("assume axis-aligned slice here corresponds to axis-aligned slice there")
            slice_params_mean_shape = (np.array((mean_shape_mesh.vertices[:,0].mean(), mean_shape_mesh.vertices[:,1].mean(), current_z_sdf)), np.array((1.,0,0)), np.array((0,1,0)), 200., 200.)

            procrustes_predictions = iterative_slice_matching(model, us_mesh, ultrasound, mean_shape_sdf, slice_params_us, num_slice_patches, num_sdf_patches,
                slice_patch_size, sdf_patch_size, z_restriction_width, kde_bandwidth, kde_k_largest, device=device, verbose=verbose, smallest_k=smallest_k, dists_thresh=thresh, sdf_batch_size=sdf_batch_size,
                                                              num_iterations=iterations, use_previous_effective_threshold=use_previous_effective_threshold,
                                                              sdf_mesh=mean_shape_mesh, hungarian_matching=hungarian_matching)
            procrustes_predictions = sorted(procrustes_predictions, key=lambda p: p[1]) # sort by loss
            # Procrustes transform says: how do I have to transform slice_params_us to make it match the matching patches in the mean shape sdf?

            procrustes_transform_slices = [transform_slice_params(slice_params_us, transform) for transform, _ in procrustes_predictions]
            # current_results.append([slices_mean_distance(slice_params, matched_slice_params) for matched_slice_params in procrustes_transform_slices])
            # print(data_index, "z =", current_z, ":", current_results[-1])
                       
            # TODO back-transfer slice!
            for candidate_index in range(len(procrustes_predictions)):
                centers_dist, angle = compute_angle_centroid_distance_metrics(procrustes_transform_slices[candidate_index],slice_params_mean_shape)
                results_list.append([data_index, current_z_us, current_z_sdf, current_z_index, candidate_index, procrustes_transform_slices[candidate_index],slice_params_mean_shape,centers_dist, angle,
                    *procrustes_predictions[candidate_index], slices_mean_distance(slice_params_mean_shape, procrustes_transform_slices[candidate_index])])

        results_df = pd.DataFrame(results_list, columns=["thyroid_index", "z_value", "z_value_mean_shape", "z_index", "candidate_index", "candidate_slice_params","gt_slice_params","centers_dist", "angle",
            "candidate_transform", "candidate_procrustes_loss", "candidate_slices_mean_distance",])
        results_df.to_csv(output_path / f"output_matching_{log_filename}")

# run_iterative_slice_matching_experiment(num_sdf_patches=300, num_slice_patches=100, slice_patch_size = np.array([32,32,32]), sdf_patch_size = np.array([32,32,32]),
#    num_slices_per_thyroid = 50, smallest_k = 50, kde_k_largest = 5, z_restriction_width = 10, kde_bandwidth = 0.05, use_previous_effective_threshold=True)

def find_soft_assignment_slice_mesh_matches(model: SdfVoxelMatchNet, slice_patches_dataset: PatchesDataset, sdf_dataset: PatchesDataset,
        device, similarity_function, embedding_dimension=128, slice_batch_size=75, sdf_batch_size=75,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    slice_dataloader = torch.utils.data.DataLoader(slice_patches_dataset, batch_size=slice_batch_size)
    sdf_dataloader = torch.utils.data.DataLoader(sdf_dataset, batch_size=sdf_batch_size)

    slice_patch_coords, sdf_patch_coords = slice_patches_dataset.patches_coords, sdf_dataset.patches_coords

    slice_embeddings, sdf_embeddings = compute_embeddings(model, slice_dataloader, sdf_dataloader, device, embedding_dimension)

    soft_assignment = soft_assignment_matrix(slice_embeddings, sdf_embeddings, model.sink_value, similarity_function)
    assignment = hard_assignment(soft_assignment)
    slice_matches, sdf_matches = assignment
    slice_match_coords, sdf_match_coords = slice_patch_coords[slice_matches,:], sdf_patch_coords[sdf_matches,:]

    return slice_matches, sdf_matches, slice_match_coords, sdf_match_coords, soft_assignment

def soft_assignment_matching(model: SdfVoxelMatchNet, ultrasound: torch.Tensor,  sdf: torch.Tensor, us_mesh: trimesh.Trimesh, sdf_mesh: trimesh.Trimesh,
        slice_params, num_sdf_patches: int, num_slice_patches: int, device: str, similarity_function, slice_patch_size=np.array([32,32,32]),
        sdf_patch_size=np.array([32,32,32]),
        slice_batch_size=75, sdf_batch_size=75, weighted_procrustes=False) -> List[Tuple[np.ndarray, float]]:
    slice_patches_dataset = create_slice_patches_dataset(us_mesh, ultrasound, slice_params, num_slice_patches, slice_patch_size, perturb_std_dev=0.0)
    sdf_patches_dataset = create_sdf_patches_dataset(sdf_mesh, sdf, num_sdf_patches, sdf_patch_size)

    _, _, slice_match_coords, sdf_match_coords, soft_assignment = find_soft_assignment_slice_mesh_matches(
        model, slice_patches_dataset, sdf_patches_dataset, device, similarity_function, slice_batch_size=slice_batch_size, sdf_batch_size=sdf_batch_size)
    if not weighted_procrustes:
        procrustes_transformation, procrustes_loss = procrustes_prediction(sdf_match_coords, slice_match_coords, return_cost=True)
    else:
        raise Exception("Weighted procrustes not implemented yet.")
        # slice_patch_coords, sdf_patch_coords = slice_patches_dataset.patches_coords, sdf_patches_dataset.patches_coords
        # slices_coords_repeat = slice_patch_coords.repeat((1, sdf_patch_coords.shape[0])).reshape(-1, slice_patch_coords.shape[1]).cpu().detach().numpy()
        # sdf_coords_repeat = sdf_patch_coords.repeat((slice_patch_coords.shape[0], 1)).reshape(-1, sdf_patch_coords.shape[1]).cpu().detach().numpy()
        # weights = soft_assignment[:-1,:-1].reshape(-1).cpu().detach().numpy()
        # procrustes_transformation, procrustes_loss = procrustes_prediction(sdf_coords_repeat, slices_coords_repeat, return_cost=True, weights=weights)

    return [(procrustes_transformation, procrustes_loss)]

def run_soft_assignment_slice_matching_experiment_meanshape(model: SdfVoxelMatchNet, num_sdf_patches: int, num_slice_patches: int, 
        model_path: str, us_data_paths: List[str], mesh_paths: List[str], mean_shape_sdf: torch.Tensor, mean_shape_mesh: trimesh.Trimesh,
        device: str, slice_patch_size=np.array([32,32,32]), sdf_patch_size=np.array([32,32,32]), num_slices_per_thyroid=50,
        slice_batch_size=75, sdf_batch_size=75, dot_product_similarity=True, weighted_procrustes=False):

# def run_iterative_slice_matching_experiment_meanshape(model, num_sdf_patches, num_slice_patches, model_path,
#         us_data_paths: List[str], mesh_paths: List[str], mean_shape_sdf: torch.Tensor, mean_shape_mesh: trimesh.Trimesh,
#         device, slice_patch_size = np.array([32,32,32]), sdf_patch_size = np.array([32,32,32]),
#         num_slices_per_thyroid = 50, smallest_k = None, thresh=None, kde_k_largest = 5, z_restriction_width = 10, kde_bandwidth = 0.05, use_previous_effective_threshold=False, thyroid_range=list(range(28)),
#         sdf_batch_size=75, iterations=2):
    results_list = []
    torch.cuda.empty_cache()

    hyperparams = [
        (None, "softAssignment"),
        ("numSdfPatches", num_sdf_patches),
        ("numSlicePatches", num_slice_patches),
        ("slicePatch", "x".join(map(str, slice_patch_size))),
        ("numSlicesPerThyroid", num_slices_per_thyroid),
        (None, "embDot" if dot_product_similarity else "embDist"),
        ("procrustes", "hardAssignment" if not weighted_procrustes else "weightedSoftAssignment"),
    ]

    similarity_function = (soft_assignment_slice_matching.dot_product_similarity if dot_product_similarity
        else soft_assignment_slice_matching.negative_distance_similarity)

    date_str = datetime.now().strftime("%b%d_%H-%M-%S")
    log_filename = "sliceMatchingExperiment_" + date_str + "_" + "_".join((key + "=" if key is not None else "") + str(value) for key, value in hyperparams if value is not None) + ".csv"
    print(len(log_filename), "-", log_filename)

    for data_index, (us_path, mesh_path) in tqdm(enumerate(zip(us_data_paths, mesh_paths))):
        ultrasound = load_nii_voxels(us_path)
        us_mesh: trimesh.Trimesh  = trimesh.load_mesh(mesh_path) # type: ignore
        
        # Define z range for slice samples: not too close to border (z index +- 20 on us, and relatively scaled +-20 on sdf)
        from_us, to_us = us_mesh.vertices[:,2].min(), us_mesh.vertices[:,2].max()
        from_sdf, to_sdf = mean_shape_mesh.vertices[:,2].min(), mean_shape_mesh.vertices[:,2].max()
        scale_factor = (to_sdf - from_sdf) / (to_us - from_us)
        from_us, to_us = from_us + 20 * scale_factor, to_us - 20 * scale_factor
        from_sdf, to_sdf = from_sdf + 20 * scale_factor, to_sdf - 20 * scale_factor

        for current_z_index, current_z_factor in tqdm(enumerate(np.linspace(0, 1, num_slices_per_thyroid))):
            current_z_us = from_us + current_z_factor * (to_us - from_us)
            current_z_sdf = from_sdf + current_z_factor * (to_sdf - from_sdf)

            slice_params_us = (np.array((us_mesh.vertices[:,0].mean(), us_mesh.vertices[:,1].mean(), current_z_us)), np.array((1.,0,0)), np.array((0,1,0)), 200., 200.)
            # TODO this is simplistic ("assume axis-aligned slice here corresponds to axis-aligned slice there")
            slice_params_mean_shape = (np.array((mean_shape_mesh.vertices[:,0].mean(), mean_shape_mesh.vertices[:,1].mean(), current_z_sdf)), np.array((1.,0,0)), np.array((0,1,0)), 200., 200.)

            procrustes_predictions = soft_assignment_matching(model, ultrasound, mean_shape_sdf, us_mesh, mean_shape_mesh, slice_params_us,  num_sdf_patches, num_slice_patches,
                device, similarity_function, slice_patch_size, sdf_patch_size, slice_batch_size, sdf_batch_size, weighted_procrustes=weighted_procrustes)

            procrustes_predictions = sorted(procrustes_predictions, key=lambda p: p[1]) # sort by loss
            procrustes_transform_slices = [transform_slice_params(slice_params_us, transform) for transform, _ in procrustes_predictions]
            # current_results.append([slices_mean_distance(slice_params, matched_slice_params) for matched_slice_params in procrustes_transform_slices])
            # print(data_index, "z =", current_z, ":", current_results[-1])
            for candidate_index in range(len(procrustes_predictions)):
                results_list.append([data_index, current_z_us, current_z_sdf, current_z_index, candidate_index, procrustes_transform_slices[candidate_index],
                    *procrustes_predictions[candidate_index], slices_mean_distance(slice_params_mean_shape, procrustes_transform_slices[candidate_index])])

        results_df = pd.DataFrame(results_list, columns=["thyroid_index", "z_value", "z_value_mean_shape", "z_index", "candidate_index", "candidate_slice_params",
            "candidate_transform", "candidate_procrustes_loss", "candidate_slices_mean_distance"])
        results_df.to_csv(f"output_matching/{log_filename}")
