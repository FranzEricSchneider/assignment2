from pytorch3d.ops.knn import knn_points
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


def chamfer_loss(point_cloud_src, point_cloud_tgt):
    '''
    Implement chamfer loss as described here:
    https://github.com/UM-ARM-Lab/Chamfer-Distance-API

    knn_points documentation was very helpful:
    https://pytorch3d.readthedocs.io/en/latest/modules/ops.html#pytorch3d.ops.knn_points

    Args:
        point_cloud_src: b x n_points x 3, float32 tensor with values in R3
        point_cloud_src: b x n_points x 3, float32 tensor with values in R3
    '''
    sq_dists_st, _, _ = knn_points(point_cloud_src, point_cloud_tgt, K=1)
    avg_st = torch.sum(sq_dists_st) / point_cloud_src.shape[1]

    sq_dists_ts, _, _ = knn_points(point_cloud_tgt, point_cloud_src, K=1)
    avg_ts = torch.sum(sq_dists_ts) / point_cloud_tgt.shape[1]

    return avg_st + avg_ts


def smoothness_loss(mesh_src):
    # loss_laplacian = 
    # implement laplacian smoothening loss
    return loss_laplacian