import argparse
import os
import time

import imageio
import losses
import numpy
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import dataset_location
import torch

from utils_vox import voxels_to_mesh
from viz import hzip, spinning_mesh, spinning_points


def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('-l', '--lr', default=4e-4, type=float)
    parser.add_argument('-m', '--max_iter', default=100000, type=int)
    parser.add_argument('-f', '--log_freq', default=1000, type=int)
    parser.add_argument('-t', '--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('-n', '--n_points', default=5000, type=int)
    parser.add_argument('-c', '--w_chamfer', default=1.0, type=float)
    parser.add_argument('-s', '--w_smooth', default=0.1, type=float)
    parser.add_argument('-d', '--device', default='cuda:0', type=str)
    return parser


def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(torch.nn.functional.sigmoid(voxels_src), voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        if step % args.log_freq == 0:
            print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))

    kwargs = {"device": args.device, "dist": 2}
    gt_images = spinning_mesh(*voxels_to_mesh(voxels_tgt[0], isovalue=0), **kwargs)
    pred_images = spinning_mesh(*voxels_to_mesh(voxels_src[0], isovalue=0), **kwargs)
    imageio.mimsave("assignment2_q1_p1.gif", hzip(gt_images, pred_images), fps=20)
    print('Saved!')


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        if step % args.log_freq == 0:
            print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))

    kwargs = {"device": args.device, "dist": 1}
    gt_images = spinning_points(pointclouds_src[0], **kwargs)
    pred_images = spinning_points(pointclouds_tgt[0], **kwargs)
    imageio.mimsave("assignment2_q1_p2.gif", hzip(gt_images, pred_images), fps=20)
    print('Saved!')


def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(
        mesh_src.verts_packed().shape,
        requires_grad=True,
        device=args.device,
    )
    optimizer = torch.optim.Adam([deform_vertices_src], lr=args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        if step % args.log_freq == 0:
            print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))

    mesh_src.offset_verts_(deform_vertices_src)

    kwargs = {"device": args.device, "dist": 1}
    gt_images = spinning_mesh(mesh_tgt.verts_list()[0],
                              mesh_tgt.faces_list()[0],
                              **kwargs)
    pred_images = spinning_mesh(mesh_src.verts_list()[0],
                                mesh_src.faces_list()[0],
                                **kwargs)
    imageio.mimsave("assignment2_q1_p3.gif", hzip(gt_images, pred_images), fps=20)
    print('Saved!')


def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    feed = r2n2_dataset[0]

    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()


    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,
                                requires_grad=True,
                                device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        fit_voxel(voxels_src, voxels_tgt, args)

    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)

    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        fit_mesh(mesh_src, mesh_tgt, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
