from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils import data
import pathlib

import numpy as np
import math
import random
import scipy.io
from dgl.geometry import farthest_point_sampler
import time
import sys
sys.path.append("../")
from load_nii_open3d import load_nii_voxels_segthy, load_nii_voxels
import trimesh


    

def get_patch_coords(num_patches, ssm_vertices: torch.Tensor, us_mesh_vertices: torch.Tensor,
        us_grid: torch.Tensor, 
        us_patch_shape = (32,32,32), sdf_patch_shape = (32, 32, 32), 
        negative_sample_from_p_furthest: float=0.2
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function takes an SSM and a US+mesh sample and generates num_patches patch coordinates

    USshape : shape of US data
    USmeshvert : corresponding US coords
    ssm_vertices: tensor of shape (n, 3)
    us_mesh_vertices: tensor of shape (n, 3) that are in correspondence with ssm_vertices
    """

    us_patch_shape = torch.tensor(us_patch_shape)
    us_grid_shape = torch.tensor(us_grid.shape)
    sdf_patch_shape = np.asarray(sdf_patch_shape)
    
    ### Extract the us_mesh vertices which are not too close to the grid bounday to be a patch center
    valid_us_points_idx = torch.where(((torch.floor(us_mesh_vertices - us_patch_shape/2) >= 0) 
                        & (torch.ceil(us_mesh_vertices + us_patch_shape/2) < us_grid_shape)).all(dim=1))[0] 
    valid_points = us_mesh_vertices[valid_us_points_idx, :]
    valid_points_sampled_idx = farthest_point_sampler(valid_points.view(1,-1,3), num_patches).squeeze(0)
    pos_patch_center_idx = valid_us_points_idx[valid_points_sampled_idx] 
    
    
    # Assumption: us grid and sdf grid have same shape!
    valid_sdf_points_idx = torch.where(((torch.floor(ssm_vertices - sdf_patch_shape/2) >= 0) 
                        & (torch.ceil(ssm_vertices + sdf_patch_shape/2) < us_grid_shape)).all(dim=1))[0]
    valid_sdf_points = ssm_vertices[valid_sdf_points_idx]

    # Sample the negative patch centers and create the patches
    neg_patch_center_idx = torch.zeros(num_patches, dtype=torch.int)

    # The three arrays are (n, 6) arrays: each row is (x, y, z, width, height, depth)
    us_patch_center_coords = torch.zeros((num_patches,6), dtype=int) # type: ignore
    pos_sdf_patch_center_coords = torch.zeros((num_patches,6), dtype=int) # type: ignore
    neg_sdf_patch_center_coords = torch.zeros((num_patches,6), dtype=int) # type: ignore
    
    for i in range(num_patches):
        # eucl_dist = torch.linalg.norm(us_mesh_vertices - us_mesh_vertices[pos_patch_center_idx[i]], dim=1)
        eucl_dist = torch.linalg.norm(valid_sdf_points - ssm_vertices[pos_patch_center_idx[i]], dim=1)
        idx = torch.randint(int(eucl_dist.shape[0] * negative_sample_from_p_furthest), (1,))
        neg_patch_center_idx = torch.argsort(eucl_dist, descending=True)[idx]

        # Compute the US patch coordinates
        dx, dy, dz = us_patch_shape
        xyz = torch.floor(us_mesh_vertices[pos_patch_center_idx[i]] - us_patch_shape / 2)
        us_patch_center_coords[i] = torch.tensor([xyz[0], xyz[1], xyz[2], dx, dy, dz])

        #Compute Pos, neg coord
        mdx, mdy, mdz = sdf_patch_shape
        xyz = torch.floor(ssm_vertices[pos_patch_center_idx[i]] - sdf_patch_shape / 2)
        pos_sdf_patch_center_coords[i] = torch.tensor([xyz[0], xyz[1], xyz[2], mdx, mdy, mdz])

        xyz = torch.floor(valid_sdf_points[neg_patch_center_idx] - sdf_patch_shape / 2)
        neg_sdf_patch_center_coords[i] = torch.tensor([xyz[0][0], xyz[0][1], xyz[0][2], mdx, mdy, mdz])
    
    return us_patch_center_coords, pos_sdf_patch_center_coords, neg_sdf_patch_center_coords
    

class NewSdfVoxelDataset(data.Dataset):
    def __init__(self, us_paths, sdf_paths, mesh_paths, num_patches_per_thyroid,
            # tv = "train",
            ultrasound_patches_shape=(32,32,32), sdf_patches_shape=(32,32,32), negative_sample_from_p_furthest=0.2
            ):
            
        self.us_paths = us_paths
        self.sdf_paths = sdf_paths
        meshes: List[trimesh.Trimesh] = [trimesh.load(mesh_path) for mesh_path in mesh_paths]
        self.num_thyroids = len(self.us_paths)
        
        self.ultrasound_grids: List[Optional[torch.Tensor]] = [None for _ in us_paths]
        self.sdf_grids: List[Optional[torch.Tensor]] = [None for _ in sdf_paths]
        self.loaded_us_grid_index: Optional[int] = None
        self.loaded_sdf_grid_index: Optional[int] = None
        
        self.num_patches_per_thyroid = num_patches_per_thyroid
        self.us_mesh_vertices = torch.tensor(np.stack([mesh.vertices[:4990] for mesh in meshes], axis=0))
        self.negative_sample_from_p_furthest = negative_sample_from_p_furthest
        
        self.ultrasound_patches_shape = torch.tensor(ultrasound_patches_shape)
        self.sdf_patches_shape = torch.tensor(sdf_patches_shape)

        self.sample_patches()

        
    def set_patches_shapes_and_p_furthest(self, ultrasound_patches_shape, sdf_patches_shape,
            negative_sample_from_p_furthest):
        old_us_shape, old_sdf_shape, old_p_furthest = self.ultrasound_patches_shape, self.sdf_patches_shape, self.negative_sample_from_p_furthest
        self.ultrasound_patches_shape = torch.tensor(ultrasound_patches_shape)
        self.sdf_patches_shape = torch.tensor(sdf_patches_shape)
        self.negative_sample_from_p_furthest = negative_sample_from_p_furthest
        if not old_us_shape.equal(self.ultrasound_patches_shape) or not old_sdf_shape.equal(self.sdf_patches_shape) or old_p_furthest != negative_sample_from_p_furthest:
            # Resample, because sampling parameters have changed
            self.sample_patches()
        
    def sample_patches(self):
        us_patches_list, pos_sdf_patches_list, neg_sdf_patches_list, self.patch_thyroid_ssm_indices = [], [], [], []
        
        # ssm_vertices = self.ssm_vertices # torch.load(f"{self.sdf_dir}/SSMverts")
        us_mesh_vertices = self.us_mesh_vertices # torch.load(f"{self.sdf_dir}/corr_mesh_verts")

        for us_index, us_path in enumerate(self.us_paths):
            us_grid = load_nii_voxels(us_path)
            # ssm_sample_index = us_index
            #ssm_sample_index = torch.randint(len(self.sdf_paths), (1,)).item()
                
            # Sample new pairs of patches for the US and sampled SSM sample
            new_us_patches, new_pos_sdf_patches, new_neg_sdf_patches = get_patch_coords(
                num_patches = self.num_patches_per_thyroid, 
                ssm_vertices = us_mesh_vertices[us_index], 
                us_mesh_vertices = us_mesh_vertices[us_index], 
                us_grid = us_grid, 
                us_patch_shape = self.ultrasound_patches_shape, 
                sdf_patch_shape = self.sdf_patches_shape,
                negative_sample_from_p_furthest=self.negative_sample_from_p_furthest)

            us_patches_list.append(new_us_patches)
            pos_sdf_patches_list.append(new_pos_sdf_patches)
            neg_sdf_patches_list.append(new_neg_sdf_patches)
            self.patch_thyroid_ssm_indices.extend([(us_index, us_index)] * self.num_patches_per_thyroid)

        self.us_patch_center_coords = torch.vstack(us_patches_list)
        self.pos_sdf_patch_center_coords = torch.vstack(pos_sdf_patches_list)
        self.neg_sdf_patch_center_coords = torch.vstack(neg_sdf_patches_list)
        


    def __len__(self):
        return self.us_patch_center_coords.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        us_coords, sdf_pos_coords, sdf_neg_coords = self.us_patch_center_coords[idx], self.pos_sdf_patch_center_coords[idx], self.neg_sdf_patch_center_coords[idx]
        thyroid_index, ssm_index = self.patch_thyroid_ssm_indices[idx]
        ssm_index = thyroid_index

        x, y, z, dx, dy, dz = us_coords.tolist()
        
        if self.ultrasound_grids[thyroid_index] is None:
            if self.loaded_us_grid_index is not None:
                self.ultrasound_grids[self.loaded_us_grid_index] = None
            self.ultrasound_grids[thyroid_index] = torch.tensor(load_nii_voxels(self.us_paths[thyroid_index])).float()
            self.loaded_us_grid_index = thyroid_index

        us_patch = self.ultrasound_grids[thyroid_index][x:x+dx,y:y+dy,z:z+dz] # type: ignore

        
        if self.sdf_grids[ssm_index] is None:
            if self.loaded_sdf_grid_index is not None:
                self.sdf_grids[self.loaded_sdf_grid_index] = None
            self.sdf_grids[ssm_index] = torch.load(self.sdf_paths[ssm_index]).float()
            self.loaded_sdf_grid_index = ssm_index

        x, y, z, dx, dy, dz = sdf_pos_coords.tolist()
        sdf_patch_pos = self.sdf_grids[ssm_index][x:x+dx,y:y+dy,z:z+dz] # type: ignore

        #assert (dx,dy,dz) == (32,32,32)
        x, y, z, dx, dy, dz = sdf_neg_coords.tolist()
        sdf_patch_neg = self.sdf_grids[ssm_index][x:x+dx,y:y+dy,z:z+dz]
        
        #assert (dx,dy,dz) == (32,32,32)
        #assert us_patch.shape == (32,32,32)
        #assert sdf_patch_pos.shape == (32,32,32)
        #if sdf_patch_neg.shape != (32, 32, 32): print(sdf_patch_neg.shape)
        #assert sdf_patch_neg.shape == (32,32,32)
        
        return (us_patch, sdf_patch_pos, sdf_patch_neg,
                us_coords[:3] + self.ultrasound_patches_shape/2, sdf_pos_coords[:3] + self.sdf_patches_shape/2, sdf_neg_coords[:3] +  self.sdf_patches_shape/2)


    def reshuffle_samples(self):
        self.sample_patches()
            
    def get_block_sizes(self):
        return [self.num_patches_per_thyroid] * self.num_thyroids