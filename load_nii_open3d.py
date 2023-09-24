## Allows to import .nii files as pure numpy voxel grids, Open3D pointclouds, and Open3D voxel grids.

from typing import List, Optional, Tuple
import open3d as o3d
import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes 
import trimesh

def cartesian_product(*arrays: np.ndarray) -> np.ndarray:
    # https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

# Assumes the following shape format: (x, y, 1, z). Returns (x, y, z).
def load_nii_shape(path: str) -> Tuple:
    shape = nib.load(path).shape
    return (shape[0], shape[1], shape[3])

# Assumes an additional dimension of size 1 at index 2 exists,
# which this method squeezes
def load_nii_voxels(path: str) -> np.ndarray:
    # load nii file as numpy array
    return nib.load(path).get_fdata()[:, :, 0, :]

# Assumes no such 4th dimension exists
def load_nii_voxels_segthy(path: str) -> np.ndarray:
    # load segthy nii.gz file as numpy array
    return np.array(nib.load(path).dataobj)
    
def load_nii_normalize_intensity(path: str) -> np.ndarray:
    # load nii file as numpy array
    data = nib.load(path).get_fdata()[:, :, 0, :]
    data_scale = data/data.max()
    return data_scale

def load_trimesh(path:str) -> trimesh.Trimesh:
    # load nii file as trimesh, input an label file
    data = load_nii_voxels(path)
    verts, faces, normals, values = marching_cubes(data) 
    mesh = trimesh.Trimesh(verts, faces)
    return mesh

def load_o3dmesh(path: str) -> o3d.geometry.TriangleMesh :
    # load nii file as o3d triangle mesh, input an label file
    data = load_nii_voxels(path)
    verts, faces, normals, values = marching_cubes(data) 
    o3dmesh = o3d.geometry.TriangleMesh()
    o3dmesh.vertices = o3d.utility.Vector3dVector(verts)
    o3dmesh.triangles = o3d.utility.Vector3iVector(faces)
    return o3dmesh

def load_trimesh_simple(path : str, face_num : int= 10000) ->trimesh.Trimesh :
    """
    create trimesh mesh with a give number of faces
    mesh simplification done with open3d quadric decimation function
    arg
        path : filepath of nii datafile
        face_num : number of faces
    """
    o3dmesh = load_o3dmesh(path)
    mesh_simple = o3d.geometry.TriangleMesh.simplify_quadric_decimation(o3dmesh, face_num)
    mesh_simple.compute_vertex_normals()
    s_vert= np.asarray(mesh_simple.vertices)
    s_fac = np.asarray(mesh_simple.triangles)
    mesh = trimesh.Trimesh(s_vert, s_fac)
    return mesh

def open3d_pointcloud_from_voxels(voxels: np.ndarray, color_scaling: float=1/256):
    voxels = voxels * color_scaling
    grid = cartesian_product(*[np.array(range(d)) for d in voxels.shape])
    colors = np.outer(voxels.flatten(), np.array([1,1,1]))
    grid_pointcloud = o3d.geometry.PointCloud()
    grid_pointcloud.points = o3d.utility.Vector3dVector(grid)
    grid_pointcloud.colors = o3d.utility.Vector3dVector(colors)
    return grid_pointcloud

def load_nii_as_pointcloud(path: str):
    data_load = load_nii_voxels(path)
    verts, faces, normals, values = marching_cubes(data_load)
    verts = transform_affine(data_load.affine, verts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    return pcd

def load_nii_as_open3d_pointcloud(path: str,
    xyz_ranges: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]=None,
    color_scaling: float=1/256
) -> o3d.geometry.PointCloud:
    voxels = load_nii_voxels(path)
    if xyz_ranges is not None:
        voxels = voxels[xyz_ranges[0][0]:xyz_ranges[0][1], xyz_ranges[1][0]:xyz_ranges[1][1], xyz_ranges[2][0]:xyz_ranges[2][1]]
    return open3d_pointcloud_from_voxels(voxels, color_scaling=color_scaling)

def load_nii_as_open3d_voxelgrid(path: str,
    xyz_ranges: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]=None,
    color_scaling: float=1/256
) -> o3d.geometry.VoxelGrid:
    pointcloud = load_nii_as_open3d_pointcloud(path, xyz_ranges, color_scaling)
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pointcloud, 1)

def visualize_o3dmesh(mesh : o3d.geometry.TriangleMesh):
    mesh.comput_vertex_normals()
    o3d.visualization.draw_geometries([mesh_simple])

def plot_voxel_grid(voxels, color_scaling=3/256, as_pointcloud=True):
    pointcloud = open3d_pointcloud_from_voxels(voxels, color_scaling=color_scaling)
    if as_pointcloud:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pointcloud)
        vis.get_render_option().point_size = 15
        vis.run()
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries([o3d.geometry.VoxelGrid.create_from_point_cloud(pointcloud, 1)])
        

