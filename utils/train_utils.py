import numpy as np
from sklearn.neighbors import KDTree
import open3d as o3d
import torch

def sample_first_frame_then_sequential(dataset, scene, opt, viewpoint_stack, iteration, final_iter):
    if iteration <= dataset.sampling_first_frame_change:
        sampled_cam_no = np.random.choice(range(len(viewpoint_stack) // scene.maxtime), size=opt.batch_size)
        sampled_frame_no = 0
        sampled_frame_no = np.full_like(sampled_cam_no, sampled_frame_no)
        viewpoint_cams = [viewpoint_stack[c * scene.maxtime + f] for c, f in zip(sampled_cam_no, sampled_frame_no)]
        return sampled_frame_no, viewpoint_cams

    number_of_iterations = final_iter - dataset.sampling_first_frame_change
    # Minus frame=0, because we trained with it in the first dataset.sampling_first_frame_change iterations
    number_of_frames = scene.maxtime - 1
    frame_changing_after = number_of_iterations // number_of_frames

    iteration -= dataset.sampling_first_frame_change

    sampled_cam_no = np.random.choice(range(len(viewpoint_stack) // scene.maxtime), size=opt.batch_size)
    # We need to subtract 1 from iteration, because we start with iteration=1
    sampled_frame_no = (iteration - 1) // frame_changing_after
    sampled_frame_no += 1
    if sampled_frame_no >= number_of_frames + 1:
        # If number_of_iterations // number_of_frames doesn't divide perfectly, we remain on the last frame on the remainder iterations
        sampled_frame_no = number_of_frames
    sampled_frame_no = np.full_like(sampled_cam_no, sampled_frame_no)
    viewpoint_cams = [viewpoint_stack[c * scene.maxtime + f] for c, f in zip(sampled_cam_no, sampled_frame_no)]

    return sampled_frame_no, viewpoint_cams


def sample_sequential_frame_n_camera(scene, opt, viewpoint_stack, iteration, final_iter, is_sample_from_past: bool):
    number_of_iterations = final_iter
    number_of_frames = scene.maxtime
    frame_changing_after = number_of_iterations // number_of_frames

    sampled_cam_no = np.random.choice(range(len(viewpoint_stack) // scene.maxtime), size=opt.batch_size)
    # We need to subtract 1 from iteration, because we start with iteration=1
    sampled_frame_no = (iteration - 1) // frame_changing_after
    if sampled_frame_no >= number_of_frames:
        # If number_of_iterations // number_of_frames doesn't divide perfectly, we remain on the last frame on the remainder iterations
        sampled_frame_no = number_of_frames - 1

    # we also want to sample from past frames
    if is_sample_from_past and sampled_frame_no != 0:
        if iteration % 2 == 0:
            sampled_frame_no = np.random.randint(0, sampled_frame_no)

    sampled_frame_no = np.full_like(sampled_cam_no, sampled_frame_no)
    viewpoint_cams = [viewpoint_stack[c * scene.maxtime + f] for c, f in zip(sampled_cam_no, sampled_frame_no)]

    return sampled_frame_no, viewpoint_cams

def sample_frame_with_preference(scene, opt, dataset, viewpoint_stack):
    total_num_frames = scene.maxtime
    sampled_cam_no = np.random.choice(range(len(viewpoint_stack) // scene.maxtime), size=opt.batch_size)
    if np.random.random() < dataset.frame_preference_probability:
        sampled_frame_no = np.random.choice(dataset.frame_indices_higher_preference, size=opt.batch_size)
    else:
        sampled_frame_no = np.random.choice(range(total_num_frames), size=opt.batch_size)
    viewpoint_cams = [viewpoint_stack[c * scene.maxtime + f] for c, f in zip(sampled_cam_no, sampled_frame_no)]
    return sampled_frame_no, viewpoint_cams

def compute_closest_distances_2_gaussians(gaussian_positions, point_cloud_path):
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    pcd_vertices = np.asarray(pcd.points).astype("f")
    
    sampled_indices = np.random.choice(len(pcd_vertices), 4000, replace=False)

    tree = KDTree(gaussian_positions) # points to search within
    dist, _ = tree.query(pcd_vertices[sampled_indices], k=1) # Find closest point in gaussian_points for each point in the point cloud

    return dist

def compute_closest_distances_2_gaussians_tensor(gaussian_positions_tensor, point_cloud_path):
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    pcd_vertices_tensor = torch.tensor(np.asarray(pcd.points).astype("f"))

    pcd_indices = torch.randperm(len(pcd_vertices_tensor))[:4000]

    pcd_vertices_tensor_exp = pcd_vertices_tensor[pcd_indices].unsqueeze(1).cuda()
    gaussian_positions_tensor_exp = gaussian_positions_tensor.unsqueeze(0)

    distances = torch.sum((pcd_vertices_tensor_exp - gaussian_positions_tensor_exp) ** 2, dim=2)
    closest_distances = torch.sqrt(torch.min(distances, dim=1).values)

    return closest_distances