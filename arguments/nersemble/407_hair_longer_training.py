_base_ = './default.py'

ModelParams = dict(
    disable_filter3D=False,
    kernel_size = 0.0,
)

ModelHiddenParams = dict(
    total_num_frames = 147,
)
OptimizationParams = dict(
    maxtime = 147,
    iterations = 160_000,
    densify_until_iter = 160_000,
    position_lr_max_steps = 160_000,
    deformation_lr_max_steps = 160_000,

    radegs_regularization_from_iter = 15_000,
    max_number_gaussians = 200_000
)