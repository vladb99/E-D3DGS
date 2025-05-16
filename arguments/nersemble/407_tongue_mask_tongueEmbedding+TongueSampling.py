_base_ = './default.py'

ModelParams = dict(
    disable_filter3D=False,
    kernel_size = 0.0,
)

ModelHiddenParams = dict(
    total_num_frames = 105,
)
OptimizationParams = dict(
    maxtime = 105,
    iterations = 80_000,
    densify_until_iter = 80_000,
    position_lr_max_steps = 80_000,
    deformation_lr_max_steps = 80_000,

    radegs_regularization_from_iter = 15_000,

    frame_indices_higher_preference = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
    frame_preference_probability = 0.5,

    tongue_mask_loss_enabled = True,
    colmap_supervision_enabled = False,

    max_number_gaussians=160_000, #Added after first try resulted in OOM with ca. 172K gaussians
)