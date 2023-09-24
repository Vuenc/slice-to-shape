import numpy as np
import torch
import open3d as o3d
import pymeshlab
import trimesh
import scipy.io
import pysdf
from pathlib import Path

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


def SSMsample2sdf(numSSMsamp, train_files, outdir, grid_bound=70, std=0.5):
    print("Generating SSM sampling..")
    ### SSM sample generation
    _, _, _, verts  = eigenshapes(train_files)
    ssm_class=SSM(verts=verts)
    samples=ssm_class.generate_random_samples(num_samples = numSSMsamp,std = std)  
    
    ### get traslation for SSM samples
    ty= np.zeros((0,3))
    vert_t = []
    
    for i, file in enumerate(train_files):
        mat = scipy.io.loadmat(file)
        ty = np.vstack([ty, mat['translation_y']])
        vert_t.append(verts[:,:,i] + mat['translation_y'])   ### translated mesh vertices (5000, 3, 28)
        
    vert_t = np.asarray(vert_t)
    tavg = np.average(ty, axis = 0)

    shift_samples = samples + tavg   # translate SSM samples (num_samples, 5000, 3)
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
        torch.save(torch.tensor(sdf), outdir / f"SSMsample_{i}_sdf")

    torch.save(torch.tensor(vert_t), outdir / "corr_mesh_verts")
    torch.save(torch.tensor(SSMshift), outdir / "SSMverts")
    
    

def SSMmean2sdf(train_files, base_files, outdir, grid_bound = 70):
    ### SSM sample generation
    _, _, mean, verts  = eigenshapes(train_files)
    samples = mean.reshape(1,-1,3)  
    
    ### get traslation for SSM samples
    ty= np.zeros((0,3))
    for filename in train_files:
        mat = scipy.io.loadmat(filename)
        ty = np.vstack([ty, mat['translation_y']])
    tavg = np.average(ty, axis = 0)

    shift_samples = samples + tavg   # translate SSM samples (num_samples, 5000, 3)

    SSMshift = np.copy(shift_samples)
    
    vert_t2 = []
    _, _, _, verts2  = eigenshapes(base_files)
    count = 0
    for filename in base_files:
        mat2 = scipy.io.loadmat(filename)
        vert_t2.append(verts2[:,:,count] + mat2['translation_y'])   ### translated mesh vertices (5000, 3, 28)
        count +=1
    vert_t2 = np.asarray(vert_t2)
    
    for i in range(1):
        SSMshift[i] = shift_samples[i] - np.amin(shift_samples[i],0 ) + grid_bound   ### make smallest coord to (40, 40, 40)
        SSMmesh = pcd2mesh(SSMshift[i])
        SSMmesh.export(outdir / "meanshape.ply")
        grid = (np.amax(SSMshift[i], 0) + grid_bound).astype(int)   # grid for sdf
        xp = np.arange(0, grid[0])
        yp = np.arange(0, grid[1])
        zp = np.arange(0, grid[2])
        points = cartesian_product(xp, yp, zp)
        
        f = pysdf.SDF(SSMmesh.vertices, SSMmesh.faces)
        sdf_at_points = f(points)
        sdf = sdf_at_points.reshape(grid)
        torch.save(torch.tensor(sdf), outdir / f"SSMsample_{i}_sdf")

    torch.save(torch.tensor(vert_t2), outdir / "corr_mesh_verts")
    torch.save(torch.tensor(SSMshift), outdir / "SSMverts")


    
def SSMmean2sdf_foronlymean(basedir_train, basedir , outdir, thyidx_start, thyidx_end, grid_bound = 70):
    ### SSM sample generation
    vals, vecs, mean, verts  = eigenshapes(basedir_train)
    #ssm_class=SSM(verts=verts)
    samples = mean.reshape(1,-1,3)  
    
    vals, vecs, mean, verts  = eigenshapes(basedir)
    ### get traslation for SSM samples
    ty= np.zeros((0,3))
    vert_t = []
    for i in range(thyidx_start, thyidx_end):    ### Original dataset
        mat = scipy.io.loadmat(basedir + '/out_0_'+str(2*i)+'.mat')  ## loading from 0_0 till 0_30   ## change 2 * i to i when using segthy
        ty = np.vstack([ty, mat['translation_y']])
        vert_t.append(verts[:,:,i] + mat['translation_y'])   ### translated mesh vertices (5000, 3, 28)
    vert_t = np.asarray(vert_t)
    tavg = np.average(ty, axis = 0)

    shift_samples = samples + tavg   # translate SSM samples (num_samples, 5000, 3)

    SSMshift = np.copy(shift_samples)
    
    for i in range(1):
        SSMshift[i] = shift_samples[i] - np.amin(shift_samples[i],0 ) + grid_bound   ### make smallest coord to (40, 40, 40)
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

    
def generatemeanshape(basedir, outdir, val_idx, ref_idx , grid_bound = 70):
    ### SSM sample generation
    _, _, mean, verts  = eigenshapes(basedir, val_idx)
    mean_vert = mean.reshape(1,-1,3)
    
    tot = np.arange(0,16)
    val = np.arange(val_idx[0],val_idx[1])
    trainidx = np.delete(tot, val)
    
    ### get traslation for SSM samples
    ty= np.zeros((0,3))
    vert_t = []
    for i in trainidx:
        mat = scipy.io.loadmat(basedir + '/out_'+str(ref_idx)+'_'+str(2*i)+'.mat') 
        ty = np.vstack([ty, mat['translation_y']])
        vert_t.append(verts[:,:,i] + mat['translation_y'])   ### translated mesh vertices (5000, 3, 28)
    vert_t = np.asarray(vert_t)
    tavg = np.average(ty, axis = 0)

    shift_samples = mean_vert + tavg   # translate SSM samples (num_samples, 5000, 3)

    SSMshift = np.copy(shift_samples)
    i = 0
    SSMshift[i] = shift_samples[i] - np.amin(shift_samples[i],0 ) + grid_bound
    SSMmesh = pcd2mesh(SSMshift[i])
    trimesh.exchange.export.export_mesh(SSMmesh, outdir+"meanmesh.ply", 'ply')
    grid = (np.amax(SSMshift[i], 0) + grid_bound).astype(int)   # grid for sdf
    xp = np.arange(0, grid[0])
    yp = np.arange(0, grid[1])
    zp = np.arange(0, grid[2])
    points = cartesian_product(xp, yp, zp)

    f = pysdf.SDF(SSMmesh.vertices, SSMmesh.faces)
    sdf_at_points = f(points)
    sdf = sdf_at_points.reshape(grid)
    torch.save(torch.tensor(sdf), outdir+"SSMsample"+str(i)+"sdf") #meanmeshsdf
    torch.save(torch.tensor(vert_t), outdir + "corr_mesh_verts") #corresponding mesh vertices
    torch.save(torch.tensor(SSMshift), outdir + "SSMverts") # meanmesh cord
    
    
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
    
