from scene import Scene
import torch
import trimesh
from utils.general_utils import build_rotation

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