import torch

# define losses
def voxel_loss(voxel_src, voxel_tgt):
    '''
    Args:
        voxel_src: b x h x w x d, float32 tensor with values 0-1
        voxel_tgt: b x h x w x d, float32 tensor with values 0-1
    '''
    loss = torch.nn.BCELoss()
    sigmoid = torch.nn.Sigmoid()
    return loss(sigmoid(voxel_src), voxel_tgt)

def chamfer_loss(point_cloud_src,point_cloud_tgt):
    # point_cloud_src, point_cloud_src: b x n_points x 3  
    # loss_chamfer = 
    # implement chamfer loss from scratch
    return loss_chamfer

def smoothness_loss(mesh_src):
    # loss_laplacian = 
    # implement laplacian smoothening loss
    return loss_laplacian