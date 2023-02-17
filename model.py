import numpy
import pytorch3d
from pytorch3d.utils import ico_sphere
from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch


class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](
                weights=torchvision_models.ResNet18_Weights.DEFAULT
            )
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 1 x 32 x 32 x 32
            self.decoder = VoxNetwork(512)
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3
            self.n_point = args.n_points
            # TODO:
            # self.decoder =
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            # self.decoder =

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            return self.decoder(encoded_feat)

        elif args.type == "point":
            # TODO:
            # pointclouds_pred =             
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            # deform_vertices_pred =             
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          


# TODO: Try paper
# Learning a predictable and generative vector representation for objects
class VoxNetwork(nn.Module):

    def __init__(self, input_size, start_channels=8, dropout=0.1):

        self.start_channels = start_channels
        self.start_3d_shape = (8, 8, 8)
        self.linear_out = self.start_channels * \
                          numpy.product(self.start_3d_shape)

        super(VoxNetwork, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.Dropout(p=dropout),
            nn.ReLU(),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(num_features=2048),
            nn.Dropout(p=dropout),
            nn.ReLU(),

            nn.Linear(2048, self.linear_out),
            nn.BatchNorm1d(num_features=self.linear_out),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )
        # TODO: Read this:
        # https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers
        # def convtrans(d, pad, dil, kern, stride, opad):
        #     return (d-1)*stride - (2*pad) + (dil*(kern-1)) + opad + 1
        # def conv(d, pad, dil, kern, stride):
        #     import math
        #     return math.floor((d + (2 * pad) - dil*(kern - 1) - 1) / stride + 1
        # )
        # self.upconvs = nn.Sequential(*(
        #     (
        #         [  # 8 > 10 > 12 > 14
        #             nn.ConvTranspose3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0),
        #             nn.BatchNorm3d(num_features=8),
        #             nn.Dropout(p=dropout),
        #             nn.ReLU(),
        #         ] * 3
        #     ) + [  # 14 > 16
        #         nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=0),
        #         nn.BatchNorm3d(num_features=4),
        #         nn.Dropout(p=dropout),
        #         nn.ReLU(),
        #     ] + (
        #         [  # 16 > 18 > 20 > 22
        #             nn.ConvTranspose3d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=0),
        #             nn.BatchNorm3d(num_features=4),
        #             nn.Dropout(p=dropout),
        #             nn.ReLU(),
        #         ] * 3
        #     ) + [  # 22 > 24
        #         nn.ConvTranspose3d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=0),
        #         nn.BatchNorm3d(num_features=2),
        #         nn.Dropout(p=dropout),
        #         nn.ReLU(),
        #     ] + (
        #         [  # 24 > 26 > 28 > 30
        #             nn.ConvTranspose3d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=0),
        #             nn.BatchNorm3d(num_features=2),
        #             nn.Dropout(p=dropout),
        #             nn.ReLU(),
        #         ] * 3
        #     ) + [  # 30 > 32
        #         nn.ConvTranspose3d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=0),
        #         nn.BatchNorm3d(num_features=1),
        #         nn.Dropout(p=dropout),
        #         nn.Sigmoid(),
        #     ]
        # ))
        self.upconvs = nn.Sequential(
            nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1), # 8 > 15
            nn.BatchNorm3d(num_features=4),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=0, output_padding=1), # 15 > 31+1
            nn.BatchNorm3d(num_features=1),
            nn.Dropout(p=dropout),
            nn.Sigmoid()
        )
        # TODO: Read the paper: https://arxiv.org/pdf/1603.08637.pdf
        # TODO: Make overfit mechanism
        # TODO: Hook it up to wandb
        # TODO: Make it DEEPER (figure out why crashing)
        # TODO: Try on AWS


    def forward(self, x):

        # Linear layers
        x = self.linear(x)

        # Reshape operation
        num_batches = x.shape[0]
        x = x.reshape((num_batches, self.start_channels) + self.start_3d_shape)

        # Upsampling
        x = self.upconvs(x)
        return x
