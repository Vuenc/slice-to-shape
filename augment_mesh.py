import numpy as np

def get_connected_vertices(vert_idxs, adj):
    '''
    get indices of vertices connected with input list of vertices
    vert_idxs : index of vertices 
    adj : adjancency matrix of mesh vertices
    return : index of vertices connected to vert_idxs
    '''
    idx = []
    for i in range(len(vert_idxs)):
        adj_idx = np.argwhere(adj[vert_idxs[i]])
        adj_idx = adj_idx.reshape(adj_idx.shape[0])
        idx = np.append(idx, adj_idx).astype(int)
    idx = np.unique(idx)
    return idx

def mesh_patch_verticies(start_idx, radius, adj):
    '''
    get vertices to modify mesh
    start_idx : center of the vertices that will be defomed
    radius : radius to deform
    return : list containing vertices around the center vertices
    '''
    list_vert = []
    tmp = np.array([start_idx]) # list of all vertices (to check if there is duplicate)
    center_vert = np.array([start_idx])
    list_vert.append(center_vert)
    for i in range(radius):
        verts = get_connected_vertices(center_vert, adj)
        verts = np.setdiff1d(verts, tmp)
        tmp = np.append(tmp, verts).astype(int)
        center_vert = verts
        list_vert.append(verts)
    return list_vert


def gaussian3d (v, mu, sig):
     #(1/(np.power(np.sqrt(2*np.pi)*sig,3.))) *
    return np.exp(-np.linalg.norm(v-mu)/(2*np.power(sig, 2.)))


def modify_vert(init_point, adj, rad, norms, vert, sig, gauss_mul, vertex_colors):
    """
    modifies input verticies according to 3d gaussian and norm of the vertex point
    perturb the vertex by f(u) * norm direction
        with f  as 3D gaussian centered at init_point
    init_point : initial vertex point index (center point of pertubation)
    rad : radius to modify
    norms : norms of the vertices
    vert : vertices of meshes
    vertex_colors : for visualization
    return : vert - returns modified vertices
    """
    vert_list = mesh_patch_verticies(init_point, rad, adj)
    add_const = []
    for i in range(len(vert_list)):
        tmp = []
        for j in range(len(vert_list[i])):
            tmp.append(np.abs(gaussian3d(vert_list[i][j], init_point, sig))*gauss_mul * norms[vert_list[i][j]] )
        add_const.append(tmp)
    vertex_colors[np.hstack(vert_list),:] = np.array([0,1.,0])
    for i in range(rad+1):
        vert[vert_list[i]] = vert[vert_list[i]] + add_const[i]
    
    return vert, vertex_colors