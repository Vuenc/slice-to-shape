import numpy as np
import torch
import open3d as o3d
import pymeshlab
import trimesh
import scipy.io
import pysdf

import sys
sys.path.append("../../Statistical Thyroid Model/Functional_maps_approach")
from eigenshapes_inputfile import eigenshapes
from eval_ssm_davies import SSM

sys.path.append("../")
from load_nii_open3d import load_trimesh_simple, cartesian_product

def pcd2mesh(pcvert):
    """
    pcvert : point cloud verticies
    returns trimesh
    """
    pcd=o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcvert))
    cl, ind1 = pcd.remove_radius_outlier(nb_points=50, radius=30)  ##works for right, rotation
    inlier_cloud = pcd.select_by_index(ind1)
    pcvert = np.asarray(inlier_cloud.points)
    mesh = pymeshlab.pmeshlab.Mesh(pcvert)
    ms = pymeshlab.pmeshlab.MeshSet()
    ms.add_mesh(mesh)
    ms.compute_normal_for_point_clouds()
    ms.apply_normal_point_cloud_smoothing()
    ms.generate_surface_reconstruction_screened_poisson()
    ms.meshing_invert_face_orientation(forceflip=False)
    vertices = ms.mesh(1).vertex_matrix()
    faces = ms.mesh(1).face_matrix()
    face_normals = ms.mesh(1).face_normal_matrix()
    vertex_normals = ms.mesh(1).vertex_normal_matrix()
    mesh = trimesh.Trimesh(vertices, faces, face_normals, vertex_normals)

    ### removing isolated meshes
    ### get connected edges, from the list, only use the longest list
    tmp = trimesh.graph.connected_components(mesh.edges, min_len=3)
    a = []
    for i in range(len(tmp)):
        a.append(len(tmp[i]))
    aidx = a.index(max(a))

    mask = np.zeros(vertices.shape[0], dtype=bool)
    mask[tmp[aidx]] = True
    mesh.update_vertices(mask)
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    return mesh


def SSMsample2sdf(numSSMsamp, basedir, outdir, val_idx, ref_idx, grid_bound = 70, std = 1):
    _, _, _, verts  = eigenshapes(basedir, val_idx)
    ssm_class=SSM(verts=verts)
    samples=ssm_class.generate_random_samples(num_samples=numSSMsamp, std = std)  
    
    tot = np.arange(0, 16)
    val = np.array(val_idx)
    trainidx = np.delete(tot, val)
    
    ### get traslation for SSM samples
    
    vert_t = []

    for i in tot:   
        mat = scipy.io.loadmat(basedir + '/out_0_'+str(i)+'.mat')  
        vert_t.append(verts[:,:,i] + mat['translation_y'])      
    vert_t = np.asarray(vert_t)
    
    ty= np.zeros((0,3))
    for i in trainidx:    ### Original dataset
        mat = scipy.io.loadmat(basedir + '/out_' + str(ref_idx)+'_'+str(i)+'.mat')  
        ty = np.vstack([ty, mat['translation_y']])
    tavg = np.average(ty, axis = 0)

    shift_samples = samples + tavg  
    SSMshift = np.copy(shift_samples)
    
    for i in range(numSSMsamp):
        SSMshift[i] = shift_samples[i] - np.amin(shift_samples[i],0 ) + grid_bound   
        SSMmesh = pcd2mesh(SSMshift[i])
        grid = (np.amax(SSMshift[i], 0) + grid_bound).astype(int)   
        xp = np.arange(0, grid[0])
        yp = np.arange(0, grid[1])
        zp = np.arange(0, grid[2])
        points = cartesian_product(xp, yp, zp)
        
        f = pysdf.SDF(SSMmesh.vertices, SSMmesh.faces)
        sdf_at_points = f(points)
        sdf = sdf_at_points.reshape(grid)
        torch.save(torch.tensor(sdf), outdir+"SSMsample"+str(i)+"sdf")

    torch.save(torch.tensor(vert_t), outdir + "corr_mesh_verts")
    torch.save(torch.tensor(SSMshift), outdir + "SSMverts") 
    
    

    
def SSMmean2sdf(basedir, outdir, val_idx, ref_idx, grid_bound = 70):
    ### SSM sample generation
    _, _, mean, verts  = eigenshapes(basedir, val_idx)
    mean_vert = mean.reshape(1,-1,3)  
    
    tot = np.arange(0, 16)
    val = np.array(val_idx)
    trainidx = np.delete(tot, val)
    
    ### get traslation for SSM samples
    ty= np.zeros((0,3))
    
    for i in trainidx:    ### Original dataset
        mat = scipy.io.loadmat(basedir + '/out_' + str(ref_idx)+'_'+str(i)+'.mat')  
        ty = np.vstack([ty, mat['translation_y']])
    tavg = np.average(ty, axis = 0)

    shift_samples = mean_vert + tavg  
    SSMshift = np.copy(shift_samples)
    
    vert_t = []
    for i in tot:
        mat = scipy.io.loadmat(basedir + '/out_' + str(ref_idx)+'_'+str(i)+'.mat')  
        vert_t.append(verts[:,:,i] + mat['translation_y'])
    vert_t = np.asarray(vert_t)  
    
    
    for i in range(1):
        SSMshift[i] = shift_samples[i] - np.amin(shift_samples[i],0 ) + grid_bound   
        SSMmesh = pcd2mesh(SSMshift[i])
        grid = (np.amax(SSMshift[i], 0) + grid_bound).astype(int)   # grid for sdf
        xp = np.arange(0, grid[0])
        yp = np.arange(0, grid[1])
        zp = np.arange(0, grid[2])
        points = cartesian_product(xp, yp, zp)
        f = pysdf.SDF(SSMmesh.vertices, SSMmesh.faces)
        sdf_at_points = f(points)
        sdf = sdf_at_points.reshape(grid)
        torch.save(torch.tensor(sdf), outdir+"SSMsample"+str(i)+"sdf")

    torch.save(torch.tensor(vert_t), outdir + "corr_mesh_verts")
    torch.save(torch.tensor(SSMshift), outdir + "SSMverts")  
    
    
def generateSDForidataset(label_dir, outdir, start , end ,grid_bound = 70):
    tot_vert = np.zeros((16, 4950, 3))
    shift_mesh_verts = np.zeros((16, 4950, 3))
    
    for i in range(start, end):
        mesh_vert = load_trimesh_simple(label_dir + str(2*i) + '-labels.nii').vertices
        mesh_vert = np.array(mesh_vert[:4950])
        tot_vert[i] = mesh_vert

        meshshift = mesh_vert - np.amin(mesh_vert, 0) + grid_bound
        shift_mesh_verts[i] = meshshift

    for i in range(start, end):
        SSMmesh = pcd2mesh(shift_mesh_verts[i])
        grid = (np.amax(shift_mesh_verts[i], 0) + grid_bound).astype(int)   # grid for sdf
        xp = np.arange(0, grid[0])
        yp = np.arange(0, grid[1])
        zp = np.arange(0, grid[2])
        points = cartesian_product(xp, yp, zp) 
        f = pysdf.SDF(SSMmesh.vertices, SSMmesh.faces)
        sdf_at_points = f(points)
        sdf = sdf_at_points.reshape(grid)
        torch.save(torch.tensor(sdf), outdir+str(i)+"sdf")

    torch.save(torch.tensor(tot_vert), outdir + "corr_mesh_verts")
    torch.save(torch.tensor(shift_mesh_verts), outdir + "SSMverts") 
    
