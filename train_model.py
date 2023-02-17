import argparse
import gc
import json
from pathlib import Path
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
import time
import torch
from torchsummaryX import summary
import wandb

import dataset_location
import losses
from model import SingleViewto3D
from r2n2_custom import R2N2


# python train_model.py --type vox --save_freq 2000 --log_freq 100 --batch_size 4 --num_workers 8
def get_args_parser():
    parser = argparse.ArgumentParser("Singleto3D", add_help=False)
    # Model parameters
    parser.add_argument("-n", "--name", default="run", help="Human readable name for this run")
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_iter", default=10000, type=int)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--type", default="vox", choices=["vox", "point", "mesh"], type=str)
    parser.add_argument("--n_points", default=5000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=0.1, type=float)
    parser.add_argument("--save_freq", default=2000, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--load_feat", action="store_true")
    parser.add_argument("--load_checkpoint", action="store_true")
    return parser


def preprocess(feed_dict,args):
    images = feed_dict['images'].squeeze(1)
    if args.type == "vox":
        voxels = feed_dict['voxels'].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        mesh = feed_dict['mesh']
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)
        ground_truth_3d = pointclouds_tgt
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]
    if args.load_feat:
        feats = torch.stack(feed_dict['feats'])
        return feats.to(args.device), ground_truth_3d.to(args.device)
    else:
        return images.to(args.device), ground_truth_3d.to(args.device)


def calculate_loss(predictions, ground_truth, args):
    if args.type == 'vox':
        loss = losses.voxel_loss(predictions,ground_truth)
    elif args.type == 'point':
        loss = losses.chamfer_loss(predictions, ground_truth)
    elif args.type == 'mesh':
        sample_trg = sample_points_from_meshes(ground_truth, args.n_points)
        sample_pred = sample_points_from_meshes(predictions, args.n_points)

        loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
        loss_smooth = losses.smoothness_loss(predictions)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth
    return loss


def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    train_loader = iter(loader)

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.train()

    # ============ print model information ... ============
    summary_dict = next(train_loader)
    summary_ims, _ = preprocess(summary_dict, args)
    model_summary = summary(model, summary_ims.to(args.device), args)
    print(model_summary)

    # ============ print model information ... ============
    # Garbage collect
    torch.cuda.empty_cache()
    gc.collect()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['step']
        print(f"Succesfully loaded iter {start_iter}")

    # ============ prepare reporter ... ============
    keyfile = Path("/workspace/wandb.json")
    assert keyfile.is_file(), \
           f"Need to populate {keyfile} with json containing wandb key"
    wandb.login(key=json.load(keyfile.open("r"))["key"])
    run = wandb.init(
        name=args.name,
        project="l43d",
        entity="franzericschneider",
        config=vars(args),
    )
    # Save the model
    with open("model_arch.txt", "w") as arch_file:
        file_write = arch_file.write(str(model))
    wandb.save("model_arch.txt")

    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0: #restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()
        feed_dict = next(train_loader)
        images_gt, ground_truth_3d = preprocess(feed_dict,args)
        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, args)

        loss = calculate_loss(prediction_3d, ground_truth_3d, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        if (step % args.save_freq) == 0:
            file = f"checkpoint_{args.type}.pth"
            torch.save({"step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                       file)
            wandb.save(file)

        if (step % args.log_freq) == 0:
            print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (step, args.max_iter, total_time, read_time, iter_time, loss.cpu().item()))
            wandb.log({
                "lr": optimizer.param_groups[0]["lr"],  # TODO: Update if scheduler
                "train-loss": loss.cpu().item(),
            })

        del feed_dict, images_gt, ground_truth_3d, prediction_3d
        torch.cuda.empty_cache()

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
