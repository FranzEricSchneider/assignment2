# Allow headless use
import matplotlib
matplotlib.use("Agg")

from matplotlib import cm, colors, pyplot
import numpy
import pytorch3d
from pytorch3d.renderer import (HardPhongShader,
                                MeshRenderer,
                                MeshRasterizer,
                                RasterizationSettings)
import torch


def get_mesh_renderer(image_size=256, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.
    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def hzip(im1s, im2s):
    return [numpy.hstack(im12) for im12 in zip(im1s, im2s)]


def spinning_mesh(verts, faces, device, image_size=256, num_views=60, dist=4,
                  elev=0):

    # For now, enforce that we are given a single mesh (it will be unsqueezed
    # later)
    assert len(verts.shape) == 2
    assert len(faces.shape) == 2

    # Colormap
    norms = numpy.linalg.norm(verts, axis=1)
    scalarmap = cm.ScalarMappable(
        norm=colors.Normalize(vmin=norms.min(), vmax=norms.max()),
        cmap=pyplot.get_cmap("gist_rainbow"), # jet, plasma
    )
    texture_rgb = torch.tensor(
        numpy.array([scalarmap.to_rgba(norm)[:3] for norm in norms]),
        dtype=torch.float32,
    )
    textures = pytorch3d.renderer.TexturesVertex(texture_rgb.unsqueeze(0))

    # Make Meshes object
    mesh = pytorch3d.structures.Meshes(
        verts.unsqueeze(0),
        faces.unsqueeze(0),
        textures=textures
    ).to(device)

    # Cameras
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=dist,
        elev=elev,
        azim=numpy.linspace(-180, 180, num_views, endpoint=False),
    )
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    lights = pytorch3d.renderer.lighting.AmbientLights(
        ambient_color=[[1, 1, 1]],
        device=device,
    )

    renderer = get_mesh_renderer(image_size=image_size, device=device)
    return [
        (image.detach().cpu().numpy()[:, :, :3] * 255).astype(numpy.uint8)
        for image in renderer(mesh.extend(num_views),
                              cameras=cameras,
                              lights=lights)
    ]
