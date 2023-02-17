# Allow headless use
import matplotlib
matplotlib.use("Agg")

from matplotlib import cm, colors, pyplot
import numpy
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
import torch


def get_points_renderer(image_size=256, device=None, radius=0.01,
                        background_color=(1,1,1)):
    """
    Returns a Pytorch3D renderer for point clouds.
    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.

    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


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

    if verts.device != "cpu":
        verts = verts.detach().cpu()
    if faces.device != "cpu":
        faces = faces.detach().cpu()

    # Empty mesh, render white
    if len(verts) == 0:
        return [
            numpy.ones((image_size, image_size, 3), dtype=numpy.uint8) * 255
            for _ in range(num_views)
        ]

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

    # Lights
    lights = pytorch3d.renderer.lighting.AmbientLights(
        ambient_color=[[1, 1, 1]],
        device=device,
    )

    renderer = get_mesh_renderer(image_size=image_size, device=device)
    return numpy_images(
        renderer(
            mesh.extend(num_views),
            cameras=spin_cams(num_views, dist, elev, device),
            lights=lights,
        )
    )


def spinning_points(points, device, image_size=256, num_views=60, dist=4,
                    elev=0):

    # For now, enforce that we are given a single cloud (it will be unsqueezed
    # later)
    assert len(points.shape) == 2

    # Get to the CPU
    if points.device != "cpu":
        points = points.detach().cpu()

    # Colormap
    norms = numpy.linalg.norm(points, axis=1)
    scalarmap = cm.ScalarMappable(
        norm=colors.Normalize(vmin=norms.min(), vmax=norms.max()),
        cmap=pyplot.get_cmap("gist_rainbow"), # jet, plasma
    )
    rgb = torch.tensor(
        numpy.array([scalarmap.to_rgba(norm)[:3] for norm in norms]),
        dtype=torch.float32,
    )

    # Make PointClouds object
    cloud = pytorch3d.structures.Pointclouds(
        points=points.unsqueeze(0),
        features=rgb.unsqueeze(0)
    ).to(device)

    renderer = get_points_renderer()
    return numpy_images(
        renderer(
            cloud.extend(num_views),
            cameras=spin_cams(num_views, dist, elev, device),
        )
    )


def spin_cams(num_views, dist, elev, device):
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=dist,
        elev=elev,
        azim=numpy.linspace(-180, 180, num_views, endpoint=False),
    )
    return pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)


def numpy_images(iterable):
    return [(image.detach().cpu().numpy()[:, :, :3] * 255).astype(numpy.uint8)
            for image in iterable]
