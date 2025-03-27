from scene import Scene
import torch
import trimesh
from utils.general_utils import build_rotation
from utils.extra_utils import o3d_knn
import open3d as o3d
import numpy as np

@torch.no_grad()
def get_tetra_points(rotation_not_activated, xyz, scale_after_3D_filter):
    M = trimesh.creation.box()
    M.vertices *= 2

    rots = build_rotation(rotation_not_activated)
    scale = scale_after_3D_filter * 3.  # TODO test

    # filter points with small opacity for bicycle scene
    # opacity = self.get_opacity_with_3D_filter
    # mask = (opacity > 0.1).squeeze(-1)
    # xyz = xyz[mask]
    # scale = scale[mask]
    # rots = rots[mask]

    # neighbors_sq_distances, _ = o3d_knn(xyz.detach().cpu().numpy(), 10)
    # neighbors_mean_distance = torch.from_numpy(neighbors_sq_distances.mean(axis=1))
    # _, indices = torch.topk(neighbors_mean_distance, k=int(neighbors_mean_distance.shape[0] * 0.05))
    # mask = torch.ones(xyz.shape[0], device=xyz.device)
    # mask = torch.round(mask).bool().squeeze()
    # mask[indices] = 0

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    # _, indices = pcd.remove_radius_outlier(nb_points=80, radius=0.02)
    # print(np.asarray(indices).shape)
    # mask = torch.zeros(xyz.shape[0], device=xyz.device)
    # mask = torch.round(mask).bool().squeeze()
    # mask[indices] = 1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    _, indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    mask = torch.zeros(xyz.shape[0], device=xyz.device)
    mask = torch.round(mask).bool().squeeze()
    mask[indices] = 1

    xyz = xyz[mask]
    scale = scale[mask]
    rots = rots[mask]

    vertices = M.vertices.T
    vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
    # scale vertices first
    vertices = vertices * scale.unsqueeze(-1)
    vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
    vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
    # concat center points
    vertices = torch.cat([vertices, xyz], dim=0)

    # scale is not a good solution but use it for now
    scale = scale.max(dim=-1, keepdim=True)[0]
    scale_corner = scale.repeat(1, 8).reshape(-1, 1)
    vertices_scale = torch.cat([scale_corner, scale], dim=0)
    return vertices, vertices_scale