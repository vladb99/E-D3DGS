from typing import Optional, Dict
from typing import TextIO
from typing import Union, List

import numpy as np
import pyvista as pv
import torch
import trimesh
import os
from utils.general_utils import build_scaling_rotation, build_rotation
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args, ModelHiddenParams
from argparse import ArgumentParser
from utils.general_utils import safe_state
from gaussian_renderer import GaussianModel
from scene import Scene
from os import makedirs

ObjectType = Dict[str, Union[List[np.ndarray], np.ndarray]]


def gaussians_to_mesh(
        gaussian_positions: torch.Tensor,
        gaussian_scales: torch.Tensor,
        gaussian_rotations: torch.Tensor,
        gaussian_colors: torch.Tensor,
        gaussian_opacities: torch.Tensor,
        use_spheres: bool = True,
        random_colors: bool = False,
        scale_factor: float = 1.5,
        ellipsoid_res: int = 5,
        opacity_threshold: float = 0.01,
        max_n_gaussians: Optional[int] = None,
        include_alphas: bool = False
) -> trimesh.Trimesh:
    gaussian_positions = gaussian_positions.detach().cpu().numpy()
    gaussian_colors = gaussian_colors.detach().cpu().numpy()
    gaussian_opacities = gaussian_opacities.detach().cpu().numpy()

    n_gaussians = len(gaussian_positions) if max_n_gaussians is None else max_n_gaussians

    if use_spheres:
        points = []
        faces = []
        points_count = 0
        face_count = 0
        all_vertex_colors = []

        base = trimesh.creation.icosphere(subdivisions=1)  # radius=0.5, count=16)

        rotm = build_scaling_rotation(gaussian_scales * scale_factor, gaussian_rotations).detach().cpu().numpy()
        for i in range(n_gaussians):
            if gaussian_opacities[i] >= opacity_threshold:
                points.append(base.vertices @ rotm[i, ...].T + gaussian_positions[i:i + 1, :])
                tris = base.faces
                face_count += tris.shape[0]
                faces.append(tris + points_count)
                points_count += base.vertices.shape[0]

                if random_colors:
                    sphere_color = np.random.rand(3)
                else:
                    sphere_color = gaussian_colors[i]
                if include_alphas:
                    vertex_colors = np.tile(np.concatenate([sphere_color[None, :], np.clip(gaussian_opacities[[i]], 0, 1)], axis=1),
                                            [base.vertices.shape[0], 1])
                else:
                    vertex_colors = np.tile(sphere_color[None, :], [base.vertices.shape[0], 1])
                all_vertex_colors.append(vertex_colors)

        points = np.concatenate(points, axis=0)
        all_vertex_colors = np.concatenate(all_vertex_colors, axis=0)
        faces = np.concatenate(faces, axis=0)
        combined_mesh = trimesh.Trimesh(points, faces, process=False, vertex_colors=all_vertex_colors)

    else:
        gaussian_scales = gaussian_scales.detach().cpu()
        gaussian_rotations = build_rotation(gaussian_rotations).detach().cpu().numpy()

        ellipsoids = []
        for i in tqdm(list(range(n_gaussians))):
            scale = gaussian_scales[i] * scale_factor
            ellipsoid = pv.ParametricEllipsoid(scale[0], scale[1], scale[2], center=gaussian_positions[i], u_res=ellipsoid_res, v_res=ellipsoid_res,
                                               w_res=ellipsoid_res)
            ellipsoids.append(ellipsoid)

        all_vertex_colors = []
        ellipsoid_meshes = []
        for ellipsoid, ellipsoid_center, ellipsoid_color, ellipsoid_opacity, ellipsoid_rotation in zip(ellipsoids, gaussian_positions, gaussian_colors,
                                                                                                       gaussian_opacities, gaussian_rotations):
            if ellipsoid_opacity >= opacity_threshold:
                faces_as_array = ellipsoid.faces.reshape((ellipsoid.n_cells, 4))[:, 1:]
                # tmesh = trimesh.Trimesh(ellipsoid.points, faces_as_array, process=False, vertex_colors=np.concatenate([ellipsoid_color, ellipsoid_opacity]))
                vertices = ellipsoid.points
                vertices = ((vertices - ellipsoid_center) @ ellipsoid_rotation) + ellipsoid_center
                if random_colors:
                    ellipsoid_color = np.random.rand(3)
                tmesh = trimesh.Trimesh(vertices, faces_as_array, process=False, vertex_colors=ellipsoid_color)
                all_vertex_colors.extend(tmesh.visual.vertex_colors)
                ellipsoid_meshes.append(tmesh)
        combined_mesh = trimesh.util.concatenate(ellipsoid_meshes)

    return combined_mesh


def load_obj(path: Union[str, TextIO], return_vn: bool = False) -> ObjectType:
    """Load wavefront OBJ from file. See https://en.wikipedia.org/wiki/Wavefront_.obj_file for file format details
    Args:
        path: Where to load the obj file from
        return_vn: Whether we should return vertex normals

    Returns:
        Dictionary with the following entries
            v: n-by-3 float32 numpy array of vertices in x,y,z format
            vt: n-by-2 float32 numpy array of texture coordinates in uv format
            vi: n-by-3 int32 numpy array of vertex indices into `v`, each defining a face.
            vti: n-by-3 int32 numpy array of vertex texture indices into `vt`, each defining a face
            vn: (if requested) n-by-3 numpy array of normals
    """

    if isinstance(path, str):
        with open(path, "r") as f:
            lines: List[str] = f.readlines()
    else:
        lines: List[str] = path.readlines()

    v = []
    vt = []
    vindices = []
    vtindices = []
    vn = []

    for line in lines:
        if line == "":
            break

        if line[:2] == "v ":
            v.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "vt":
            vt.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "vn":
            vn.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "f ":
            vindices.append([int(entry.split("/")[0]) - 1 for entry in line.split()[1:]])
            if line.find("/") != -1:
                vtindices.append([int(entry.split("/")[1]) - 1 for entry in line.split()[1:]])

    if len(vt) == 0:
        assert len(vtindices) == 0, "Tried to load an OBJ with texcoord indices but no texcoords!"
        vt = [[0.5, 0.5]]
        vtindices = [[0, 0, 0]] * len(vindices)

    # If we have mixed face types (tris/quads/etc...), we can't create a
    # non-ragged array for vi / vti.
    mixed_faces = False
    for vi in vindices:
        if len(vi) != len(vindices[0]):
            mixed_faces = True
            break

    if mixed_faces:
        vi = [np.array(vi, dtype=np.int32) for vi in vindices]
        vti = [np.array(vti, dtype=np.int32) for vti in vtindices]
    else:
        vi = np.array(vindices, dtype=np.int32)
        vti = np.array(vtindices, dtype=np.int32)

    out = {
        "v": np.array(v, dtype=np.float32),
        "vn": np.array(vn, dtype=np.float32),
        "vt": np.array(vt, dtype=np.float32),
        "vi": vi,
        "vti": vti,
    }

    if return_vn:
        assert len(out["vn"]) > 0
        return out
    else:
        out.pop("vn")
        return

def get_time_steps(scene: Scene) -> [float]:
    time_steps = []
    for cam in scene.getVideoCameras():
        time_steps.append(cam.time)
    return time_steps

def visualize_geometry(dataset : ModelParams, hyperparam: ModelHiddenParams, opt: OptimizationParams, iteration : int, timestep: int, max_n_gaussians: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, duration=hyperparam.total_num_frames, loader=dataset.loader, opt=opt)

        timesteps = get_time_steps(scene=scene)

        if timestep > len(timesteps):
            raise Exception("timestep must be smaller than the total number of frames")

        meshes_path = os.path.join(dataset.model_path, "gaussianMeshes", "ours_{}".format(scene.loaded_iter))
        makedirs(meshes_path, exist_ok=True)

        if timestep != -1:
            timesteps = [timestep[timestep - 1]]

        if max_n_gaussians == -1:
            max_n_gaussians = None

        print("Generating gaussian meshes")
        for index in tqdm(range(len(timesteps))):
            means3D = gaussians.get_xyz
            scales = gaussians._scaling
            rotations = gaussians._rotation
            opacity = gaussians._opacity
            shs = gaussians.get_features
            time = torch.tensor(timesteps[index]).to(means3D.device).repeat(means3D.shape[0], 1)
            means3D_final, scales_deformed, rotations_deformed, opacity_deformed, shs_final, extras = gaussians._deformation(
                means3D,
                scales,
                rotations,
                opacity,
                time,
                None, gaussians,
                None, shs,
                iter=scene.loaded_iter,
                num_down_emb_c=hyperparam.min_embeddings,
                num_down_emb_f=hyperparam.min_embeddings)
            rotations_final = gaussians.rotation_activation(rotations_deformed)
            scales_final = gaussians.scaling_activation(scales_deformed)
            opacity_final = gaussians.opacity_activation(opacity_deformed)

            mesh = gaussians_to_mesh(
                gaussian_positions=means3D_final,
                gaussian_colors=shs_final,
                gaussian_scales=scales_final,
                gaussian_opacities=opacity_final,
                gaussian_rotations=rotations_final,
                max_n_gaussians=max_n_gaussians,
            )
            mesh.export(os.path.join(meshes_path, '{0:05d}'.format(index) + ".ply"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--timestep", default=-1, type=int)
    parser.add_argument("--max_n_gaussians", default=-1, type=int)

    # import sys
    # args = parser.parse_args(sys.argv[1:])
    args = get_combined_args(parser)
    print("Rendering ", args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    visualize_geometry(model.extract(args), hyperparam.extract(args), opt.extract(args), args.iteration, args.timestep, args.max_n_gaussians)