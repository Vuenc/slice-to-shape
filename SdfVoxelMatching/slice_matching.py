from typing import List, Optional, Union, Tuple

from trimesh.base import Trimesh
import encoder
import sys
from load_nii_open3d import load_nii_voxels, load_trimesh_simple, cartesian_product
from voxelgrid_slices import compute_slice_mesh_trimesh, slice_corners, slice_gridpoints
import torch
import trimesh
import numpy as np
import plotly.express as px
import scipy
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from scipy.signal import argrelextrema
import torch.utils.data
import trimesh.intersections
import trimesh.sample
from numpy.typing import ArrayLike
from datetime import datetime
import scipy.spatial
import matplotlib.cm # color maps
import trimesh.registration
import pandas as pd
from tqdm.notebook import tqdm
import os 
from dgl.geometry import farthest_point_sampler
import scipy.optimize

class PatchesDataset(torch.utils.data.Dataset):
    def __init__(self, patches_coords, grid, patches_size):
        self.patches_coords = patches_coords
        self.grid = grid
        self.patches_size = patches_size
        
    def __len__(self):
        return self.patches_coords.shape[0]
    
    def __getitem__(self, idx):
        x,y,z = self.patches_coords[idx,:]
        dx,dy,dz = self.patches_size
        return self.grid[int(x):int(x+dx), int(y):int(y+dy), int(z):int(z+dz)]

def sample_on_plane_near_surface(mesh, slice_params, num_samples):
    intersection_lines = trimesh.intersections.mesh_plane(mesh, plane_normal=np.cross(slice_params[1], slice_params[2]), plane_origin=slice_params[0])
    line_lengths = np.linalg.norm(intersection_lines[:,0,:] - intersection_lines[:,1,:], axis=1)
    lines_to_sample = np.random.choice(intersection_lines.shape[0], num_samples, p=line_lengths/line_lengths.sum())
    ts = np.random.random_sample((num_samples, 1))
    points = intersection_lines[lines_to_sample,0,:] * ts + intersection_lines[lines_to_sample,1,:] * (1-ts)
    return points

def compute_embeddings(model: encoder.SdfVoxelMatchNet, slice_dataloader, sdf_dataloader, device, embedding_dimension):
    torch.cuda.empty_cache()
    slice_embeddings = torch.zeros(0, embedding_dimension).to(device)
    sdf_embeddings = torch.zeros(0, embedding_dimension).to(device)
    for i, (slice_batch, sdf_batch) in enumerate(zip(slice_dataloader, sdf_dataloader)):
        slice_batch, sdf_batch = slice_batch.float().to(device), sdf_batch.float().to(device)
        embs_slice = model.forward_ultrasound(slice_batch)
        embs_sdf = model.forward_sdf(sdf_batch)
        slice_embeddings = torch.vstack([slice_embeddings, embs_slice.detach()])
        sdf_embeddings = torch.vstack([sdf_embeddings, embs_sdf.detach()])
    return slice_embeddings, sdf_embeddings

def compute_matches(
        slice_embeddings, sdf_embeddings, slice_patch_coords, sdf_patch_coords, slice_patch_size, sdf_patch_size,
        only_one_match_per_patch=True, dists_thresh=None, smallest_k=None,
        verbose=True, hungarian_matching=False
    ):
    assert((dists_thresh is not None) ^ (smallest_k is not None))

    if hungarian_matching:
        assert(smallest_k is not None)
        dists: torch.Tensor = torch.norm(slice_embeddings.unsqueeze(0).cpu() - sdf_embeddings.unsqueeze(1).cpu(), dim=2).detach().cpu().numpy()
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(dists)
        smallest_k_inds = np.argsort(dists[row_ind, col_ind])
        sdf_matches, slice_matches = row_ind[smallest_k_inds], col_ind[smallest_k_inds]
        effective_k_or_threshold = dists[sdf_matches[-1], slice_matches[-1]]
    elif only_one_match_per_patch:
        tree = KDTree(sdf_embeddings.detach().cpu())
        dists, sdf_closest_points_indices = tree.query(slice_embeddings.detach().cpu())
        if dists_thresh is not None:
            slice_matches = np.where(dists < dists_thresh)[0]
            effective_k_or_threshold = len(slice_matches)
            if verbose: print("Effective k:", effective_k_or_threshold)
        else:
            # use the embedding pairs with k smallest distances
            slice_matches = np.argpartition(dists, smallest_k)[:smallest_k]
            effective_k_or_threshold = dists[slice_matches[-1]]
            if verbose: print("Effective threshold:", effective_k_or_threshold)  
        sdf_matches = sdf_closest_points_indices[slice_matches]
    else:    
        dists = torch.norm(slice_embeddings.unsqueeze(0).cpu() - sdf_embeddings.unsqueeze(1).cpu(), dim=2).detach().cpu()
        if dists_thresh is not None:
            sdf_matches, slice_matches = torch.where(dists < dists_thresh)
            effective_k_or_threshold = len(sdf_matches)
            if verbose: print("Effective k:", effective_k_or_threshold)
        else:
            # use the embedding pairs with k smallest distances
            sdf_matches, slice_matches = np.unravel_index(np.argpartition(dists.flatten(), smallest_k)[:smallest_k], dists.shape)
            effective_k_or_threshold = dists[sdf_matches[-1], slice_matches[-1]]
            if verbose: print("Effective threshold:", effective_k_or_threshold)
        
    slice_match_coords = slice_patch_coords[slice_matches,:] + slice_patch_size/2
    sdf_match_coords = sdf_patch_coords[sdf_matches,:] + sdf_patch_size/2

    # artificially restrict matches to those who are close to the center (not realistic yet like this)
    # normal_vec = torch.tensor(np.cross(slice_params[1], slice_params[2]))
    # normal_vec /= np.linalg.norm(normal_vec)
    # close_enough = torch.abs(sdf_match_coords @ normal_vec.unsqueeze(1) - torch.dot(torch.tensor(slice_params[0]), normal_vec)).squeeze(1) < 200
    # slice_match_coords = slice_match_coords[close_enough]
    # sdf_match_coords = sdf_match_coords[close_enough]
    # matches_dists = torch.norm(slice_match_coords - sdf_match_coords, dim=1).cpu()
    # px.histogram(x=matches_dists)

    return slice_matches, sdf_matches, slice_match_coords, sdf_match_coords, effective_k_or_threshold

def create_slice_patches_dataset(mesh, ultrasound_grid, slice_params, num_slice_patches, slice_patch_size, perturb_std_dev=0.0) -> PatchesDataset:
    slice_patch_candidate_coords = torch.zeros((0, 3))
    oversample_factor = 5
    
    while slice_patch_candidate_coords.shape[0] < num_slice_patches * oversample_factor:
        num_samples_remaining = num_slice_patches * oversample_factor - slice_patch_candidate_coords.shape[0]
        new_slice_patch_coords = torch.tensor(sample_on_plane_near_surface(mesh, slice_params, num_samples_remaining)
            + np.random.randn(num_samples_remaining, 3)*perturb_std_dev - slice_patch_size/2)

        good_coords = (new_slice_patch_coords >= 0).all(dim=1) & (new_slice_patch_coords + slice_patch_size - 1 <= torch.tensor(ultrasound_grid.shape)).all(dim=1)
        slice_patch_candidate_coords = torch.vstack([slice_patch_candidate_coords, new_slice_patch_coords[good_coords, :]])
        
    slice_patch_idx = farthest_point_sampler(slice_patch_candidate_coords.unsqueeze(0), num_slice_patches).squeeze(0)

    return PatchesDataset(slice_patch_candidate_coords[slice_patch_idx], ultrasound_grid, slice_patch_size)
    
def create_sdf_patches_dataset(mesh, sdf_grid, num_sdf_patches, sdf_patch_size) -> PatchesDataset:
    sdf_patch_candidate_coords = torch.zeros((0, 3))
    oversample_factor = 5
    
    while sdf_patch_candidate_coords.shape[0] < num_sdf_patches:
        num_samples_remaining = num_sdf_patches * oversample_factor - sdf_patch_candidate_coords.shape[0]
        new_sdf_patch_coords = torch.tensor(np.array(trimesh.sample.sample_surface(mesh, num_samples_remaining)[0]) - sdf_patch_size/2)
        good_coords = (new_sdf_patch_coords >= 0).all(dim=1) & (new_sdf_patch_coords + sdf_patch_size - 1 <= torch.tensor(sdf_grid.shape)).all(dim=1)
        sdf_patch_candidate_coords = torch.vstack([sdf_patch_candidate_coords, new_sdf_patch_coords[good_coords, :]])
    
    sdf_patch_idx = farthest_point_sampler(sdf_patch_candidate_coords.unsqueeze(0), num_sdf_patches).squeeze(0)

    return PatchesDataset(sdf_patch_candidate_coords[sdf_patch_idx], sdf_grid, sdf_patch_size)

def find_slice_mesh_matches(model, slice_patches_dataset, sdf_patches_dataset, device,
        dists_thresh=None, smallest_k=None, only_one_match_per_patch=True,
        embedding_dimension=128, sdf_batch_size=75, return_effective_k_or_threshold=False, verbose=False,
        hungarian_matching=False
        ):
    assert((dists_thresh is not None) ^ (smallest_k is not None))
    
    slice_batch_size = int(sdf_batch_size / len(sdf_patches_dataset) * len(slice_patches_dataset))
    slice_dataloader = torch.utils.data.DataLoader(slice_patches_dataset, batch_size=slice_batch_size)
    sdf_dataloader = torch.utils.data.DataLoader(sdf_patches_dataset, batch_size=sdf_batch_size)

    slice_patch_coords, sdf_patch_coords = slice_patches_dataset.patches_coords, sdf_patches_dataset.patches_coords
    slice_patch_size, sdf_patch_size = slice_patches_dataset.patches_size, sdf_patches_dataset.patches_size

    slice_embeddings, sdf_embeddings = compute_embeddings(model, slice_dataloader, sdf_dataloader, device, embedding_dimension)

    slice_matches, sdf_matches, slice_match_coords, sdf_match_coords, effective_k_or_threshold = compute_matches(
        slice_embeddings, sdf_embeddings, slice_patch_coords, sdf_patch_coords,
        slice_patch_size, sdf_patch_size, only_one_match_per_patch, dists_thresh, smallest_k, verbose=verbose,
        hungarian_matching=hungarian_matching)
    
    if return_effective_k_or_threshold:
        return slice_matches, sdf_matches, slice_match_coords, sdf_match_coords, effective_k_or_threshold
    return slice_matches, sdf_matches, slice_match_coords, sdf_match_coords    

def hom_coords(X):
    if len(X.shape) == 1:
        X = X[np.newaxis, :]
    return np.hstack([X, np.ones((X.shape[0], 1))])

# returns a transform that, applied to hom_coords(data2), gives approximately data1 
def procrustes_find_transform(data1, data2, scale=False, return_cost=False):
    return_data = trimesh.registration.procrustes(data2, data1, reflection=False, translation=True, scale=scale, return_cost=return_cost)
    if return_cost:
        transform = return_data[0][:3, :].T
        cost = return_data[2] if return_cost else None
        return transform, cost
    else:
        transform = return_data[:3, :].T
        return transform   


# This method has become a bit useless, but is kept for backward compatibility
def procrustes_prediction(sdf_match_coords, slice_match_coords, return_cost=False):
    return procrustes_find_transform(sdf_match_coords, slice_match_coords, return_cost=return_cost)

def cubes_mesh(points, cube_sidelength: Union[float, ArrayLike]=32., cube_colors=None):
    cube_vertices = (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]) - 0.5) * cube_sidelength
    all_vertices = (points[:,np.newaxis,:] + cube_vertices[np.newaxis,:,:]).reshape(-1, 3)
    cube_faces = np.array([[3,2,1], [1,5,4], [4,7,3], [2,6,5], [3,7,6], [6,7,4], [3,1,0], [1,4,0], [4,3,0], [2,5,1], [3,6,2], [6,4,5]])
    all_faces = (cube_faces[np.newaxis,:,:] + np.arange(points.shape[0])[:,np.newaxis,np.newaxis] * 8).reshape(-1, 3)
    face_colors = None
    if cube_colors is not None:
        face_colors = (cube_colors[:,np.newaxis,:] + np.zeros((1,12,cube_colors.shape[1]))).reshape(-1, cube_colors.shape[1])
    return trimesh.Trimesh(all_vertices, all_faces, face_colors=face_colors)

def patch_cubes_mesh(slice_match_coords: Optional[ArrayLike], sdf_match_coords: Optional[ArrayLike],
        slice_patch_size: Optional[ArrayLike], sdf_patch_size: Optional[ArrayLike]):
    assert((slice_match_coords is None) == (slice_patch_size is None))
    assert((sdf_match_coords is None) == (sdf_patch_size is None))
    assert((slice_match_coords is not None) or (sdf_match_coords is not None))

    N = slice_match_coords.shape[0] if slice_match_coords is not None else sdf_match_coords.shape[0]
    colors = np.random.random_sample((N, 3))
    slice_cubes_mesh = (cubes_mesh(slice_match_coords, slice_patch_size, cube_colors=colors)
        if slice_match_coords is not None else None)
    sdf_cubes_mesh = (cubes_mesh(sdf_match_coords, sdf_patch_size, cube_colors=colors)
        if sdf_match_coords is not None else None)
    if slice_cubes_mesh is None:
        return sdf_cubes_mesh
    elif sdf_cubes_mesh is None:
        return slice_cubes_mesh
    else:
        return slice_cubes_mesh + sdf_cubes_mesh

def transform_slice_params(slice_params, transformation_matrix):
    center, dir1, dir2, length1, length2 = slice_params
    new_center = (hom_coords(center) @ transformation_matrix).reshape(-1)
    new_dir1 = dir1 @ transformation_matrix[:3, :]
    new_dir2 = dir2 @ transformation_matrix[:3, :]
    new_length1 = np.linalg.norm(new_dir1) / np.linalg.norm(dir1) * length1
    new_length2 = np.linalg.norm(new_dir2) / np.linalg.norm(dir2) * length2
    return (new_center, new_dir1, new_dir2, new_length1, new_length2)

def show_mesh_with_slices(mesh, slices_params: List[Tuple[ArrayLike, ArrayLike, ArrayLike, float, float]],
        color_sequence=((0, 1.0, 0), (1.0, 1.0, 0)) + matplotlib.cm.Set2.colors,
        camera_angles=(-0.2, 1.15, 0), camera_distance=750, camera_center=(0, 0, 0),
        additional_meshes=[],
        return_meshes=False
    ):
    meshes = ([mesh] + [compute_slice_mesh_trimesh(*slice_params, color=color)
        for slice_params, color in zip(slices_params, color_sequence)]
        + additional_meshes)
    if not return_meshes:
        scene = trimesh.Scene(meshes)
        scene.set_camera(angles=camera_angles, distance=camera_distance, center=camera_center)
        return scene.show()
    else:
        return sum(meshes)

def kde_estimate(data, projection_matrix, show_plot=False, all_data=None, k_largest=5, kde_bandwidth: Optional[float]=0.05,
        xfrom=None, xto=None, return_x_y=False):
    proj_data = (data @ projection_matrix).reshape(-1)
    kde = gaussian_kde(proj_data)
    if kde_bandwidth is not None:
        kde.set_bandwidth(kde_bandwidth)
    if xfrom is None: xfrom = proj_data.min()
    if xto is None: xto = proj_data.max()
    x = np.linspace(xfrom, xto, 500)
    y = kde(x)

    if all_data is not None:
        # normalize density by density of all data (essentially estimating a conditional probability)
        proj_all_data = (all_data @ projection_matrix).reshape(-1)
        kde_all_data = gaussian_kde(proj_all_data)
        y = y / kde_all_data(x)

    local_extrema_inds = argrelextrema(y, np.greater)[0]
    k_largest = min(k_largest, len(local_extrema_inds))
    k_largest_inds = np.argpartition(y[local_extrema_inds], -k_largest)[-k_largest:]
    
    if show_plot:
        fig_line = px.line(x=x, y=y)
        fig_scatter = px.scatter(x=x[local_extrema_inds][k_largest_inds], y=y[local_extrema_inds][k_largest_inds], color_discrete_sequence=px.colors.qualitative.Dark2)
        go.Figure(data=fig_line.data + fig_scatter.data).show()

    # point @ projection_matrix = x
    # projection_matrix.T @ point.T - x.T = 0
    
    # for now: assume projection_matrix projects on z coordinate
    assert(np.isclose(projection_matrix, np.array([[0],[0],[1]])).all())

    candidates_order = np.argsort(-y[local_extrema_inds][k_largest_inds])
    candidates_x = x[local_extrema_inds][k_largest_inds][candidates_order]
    candidates_y = y[local_extrema_inds][k_largest_inds][candidates_order]

    if return_x_y:
        return candidates_x, candidates_y, x, y
    else:
        return candidates_x, candidates_y

def restrict_mesh_z(mesh, zmin, zmax):
    new_vertices_index = (mesh.vertices[:,2] >= zmin) & (mesh.vertices[:,2] <= zmax)
    new_faces = mesh.faces[np.where(new_vertices_index[mesh.faces].all(axis=1))]
    return trimesh.Trimesh(mesh.vertices, new_faces)

def iterative_slice_matching(model, us_mesh, ultrasound, sdf, slice_params, num_slice_patches, num_sdf_patches, slice_patch_size, sdf_patch_size,
                            z_restriction_width, kde_bandwidth, kde_k_largest, device,
                            verbose=True, dists_thresh=None, smallest_k=None, only_one_match_per_patch=True,
                            use_previous_effective_threshold=False, num_iterations=2, sdf_batch_size=75, sdf_mesh=None,
                            hungarian_matching=False):
    """
    sdf_mesh: specify if a different mesh should be used for sdf data than for us data, else leave None.
    """

    if sdf_mesh is None:
        sdf_mesh = us_mesh

    assert((dists_thresh is None) ^ (smallest_k is None))
    if use_previous_effective_threshold:
        assert(dists_thresh is None)
    
    candidate_meshes = [(None, sdf_mesh)]
    new_candidate_meshes = []
    
    procrustes_predictions = []
    
    # IDEA: use effective threshold from first iteration as fixed threshold for later iterations
    
    slice_patches_dataset = create_slice_patches_dataset(us_mesh, ultrasound, slice_params, num_slice_patches, slice_patch_size, perturb_std_dev=2.0)
    previous_effective_threshold = None # to satisfy the type checker

    for iteration in range(num_iterations):
        iteration_data_frames = []
        kde_data = []
        iteration_kde_out = np.zeros((0,2))
        for current_z, candidate_mesh in candidate_meshes:
            if verbose: print("Current Z:", current_z)
            # if iteration == 2: return candidate_mesh #.show()
            sdf_patches_dataset = create_sdf_patches_dataset(candidate_mesh, sdf, num_sdf_patches, sdf_patch_size)
            threshold_config = { # exactly one of the two first parameters can be active at the same time
                "dists_thresh": dists_thresh, # 0.11,
                "smallest_k": smallest_k, # 200,
                "only_one_match_per_patch": only_one_match_per_patch,
                "return_effective_k_or_threshold": True
            }
            if iteration > 0 and use_previous_effective_threshold:
                smallest_k = None
                dists_thresh = previous_effective_threshold
            slice_matches, sdf_matches, slice_match_coords, sdf_match_coords, previous_effective_threshold_or_k = (
                find_slice_mesh_matches(model, slice_patches_dataset, sdf_patches_dataset, device, embedding_dimension=128,
                                        sdf_batch_size=sdf_batch_size, hungarian_matching=hungarian_matching, **threshold_config))
            if iteration == 0:
                previous_effective_threshold = previous_effective_threshold_or_k
            
            procrustes_transformation, procrustes_loss = procrustes_prediction(sdf_match_coords, slice_match_coords, return_cost=True)
            procrustes_sdf_prediction = hom_coords(slice_match_coords) @ procrustes_transformation
            
            if verbose: print("Procrustes loss at z =", current_z, ":", procrustes_loss)

            procrustes_predictions.append((procrustes_transformation, procrustes_loss))

            
            z_projection = np.array([[0],[0],[1]])
            projected = sdf_match_coords @ z_projection
            # make sure that the kde_estimate does not fail due to "singular matrix"
            if projected.shape[0] == 0 or projected.min() == projected.max():
                continue
            
            # kde_data.append(sdf_match_coords)
            
            # TODOOOO    
            candidates_z, candidates_density, x, y = kde_estimate(sdf_match_coords, z_projection, show_plot=False,
                all_data = None, # sdf_patches_dataset.patches_coords, # all_data=None, # ,
                k_largest=kde_k_largest, kde_bandwidth=kde_bandwidth,
                xfrom=sdf_mesh.vertices[:,2].min(), xto=sdf_mesh.vertices[:,2].max(),
                return_x_y=True)
            # iteration_data_frames.append(pd.DataFrame({"x": x, "y": y, "z": "z = " + str(current_z)}))
            iteration_data_frames.append(pd.DataFrame({"x": x, "y": y * len(sdf_match_coords), "z": "z = " + str(current_z)}))
            iteration_kde_out = np.vstack([iteration_kde_out, np.hstack([candidates_z[:,np.newaxis], candidates_density[:,np.newaxis] * len(sdf_match_coords)])])

            #if iteration < num_iterations - 1:
            #    for z in candidates_z:
            #        new_candidate_meshes.append((z, restrict_mesh_z(mesh, z-z_restriction_width/2, z+z_restriction_width/2)))
        
        if iteration < num_iterations - 1:
            candidates_z = iteration_kde_out[np.argsort(-iteration_kde_out[:, 1])[:kde_k_largest], 0]
            for z in candidates_z:
                new_candidate_meshes.append((z, restrict_mesh_z(sdf_mesh, z-z_restriction_width/2, z+z_restriction_width/2)))
        candidate_meshes = new_candidate_meshes
        new_candidate_meshes = []
        
        if verbose: px.line(pd.concat(iteration_data_frames), x="x", y="y", color="z").show()
        
    best_procrustes_transform, best_procrustes_loss = min(procrustes_predictions, key=lambda p: p[1])

    if verbose: print("Procrustes losses:", sorted(l for _, l in procrustes_predictions))
    return procrustes_predictions

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


def run_iterative_slice_matching_experiment(model, model_path, num_sdf_patches, num_slice_patches, 
                                            us_data_paths, mesh_paths, sdf_paths, device,
                                            slice_patch_size = np.array([32,32,32]), sdf_patch_size = np.array([32,32,32]),
                                            num_slices_per_thyroid = 50, smallest_k = None, thresh=None, kde_k_largest = 5, z_restriction_width = 10, kde_bandwidth = 0.05,
                                            use_previous_effective_threshold=False, thyroid_range=list(range(28)),
                                            sdf_batch_size=75, iterations=2, only_one_match_per_patch=True, hungarian_matching=False,
                                            log_filename=None):
    results_list = []
    torch.cuda.empty_cache()

    hyperparams = [
        ("numSdfPatches", num_sdf_patches),
        ("numSlicePatches", num_slice_patches),
        ("slicePatch", "x".join(map(str, slice_patch_size))),
        ("sdfPatch", "x".join(map(str, sdf_patch_size))),
        ("numSlicesPerThyroid", num_slices_per_thyroid),
        ("smallestK", smallest_k),
        ("thresh", thresh),
        ("kdeKLargest", kde_k_largest),
        ("zRestrictionWidth", z_restriction_width),
        ("kdeBandwidth", kde_bandwidth),
        ("model", model_path[model_path.rfind("/")+1+6:model_path.rfind("/")+1+20]),
        (None, "prevEffThresh" if use_previous_effective_threshold else "noPrevEffThresh"),
        ("iter", iterations),
    ]

    date_str = datetime.now().strftime("%b%d_%H-%M-%S")
    if log_filename is None:
        log_filename = "sliceMatchingExperiment_" + date_str + "_" + "_".join((key + "=" if key is not None else "") + str(value) for key, value in hyperparams if value is not None) + ".csv"
    print(len(log_filename))
    print(log_filename)

    for data_index, (us_path, mesh_path, sdf_path) in tqdm(enumerate(zip(us_data_paths, mesh_paths, sdf_paths))):
        ultrasound = load_nii_voxels(us_path)
        mesh: trimesh.Trimesh  = trimesh.load_mesh(mesh_path) # type: ignore
        # mesh_sdf: trimesh.Trimesh  = trimesh.load_mesh(mesh_path_sdf) # type: ignore
        sdf = torch.load(sdf_path)


        for current_z_index, current_z in tqdm(enumerate(np.linspace(mesh.vertices[:,2].min() + 20, mesh.vertices[:,2].max() - 20, num_slices_per_thyroid)), leave=False):
            slice_params = (np.array((mesh.vertices[:,0].mean(), mesh.vertices[:,1].mean(), current_z)), np.array((1.,0,0)), np.array((0,1,0)), 200., 200.)
        
            
            procrustes_predictions = iterative_slice_matching(model, mesh, ultrasound, sdf, slice_params, num_slice_patches, num_sdf_patches,slice_patch_size, sdf_patch_size, 
                                                              z_restriction_width, kde_bandwidth, kde_k_largest, device, 
                                                              verbose=False, dists_thresh=thresh, smallest_k=smallest_k, only_one_match_per_patch = only_one_match_per_patch, 
                                                              use_previous_effective_threshold=use_previous_effective_threshold, num_iterations=iterations,
                                                              sdf_batch_size=sdf_batch_size, hungarian_matching=hungarian_matching)
            
            procrustes_predictions = sorted(procrustes_predictions, key=lambda p: p[1]) # sort by loss
            
            procrustes_transform_slices = [transform_slice_params(slice_params, transform) for transform, _ in procrustes_predictions]
            
            for candidate_index in range(len(procrustes_predictions)):
                centers_dist, angle = compute_angle_centroid_distance_metrics(procrustes_transform_slices[candidate_index],slice_params)
                results_list.append([data_index, current_z, current_z_index, candidate_index, 
                                     procrustes_transform_slices[candidate_index], slice_params, centers_dist, angle,
                                     *procrustes_predictions[candidate_index], slices_mean_distance(slice_params, procrustes_transform_slices[candidate_index])])

        results_df = pd.DataFrame(results_list, columns=["thyroid_index", "z_value", "z_index", "candidate_index", 
                                                         "candidate_slice_params", "gt_slice_params","centers_dist", "angle",
                                                         "candidate_transform", "candidate_procrustes_loss", "candidate_slices_mean_distance"])
    
        results_df.to_csv(os.path.join('output_matching',log_filename))