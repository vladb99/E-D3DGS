_base_ = './default.py'

ModelParams = dict(
    disable_filter3D=True,
    sequential_frame_sampling = True
)

ModelHiddenParams = dict(
    total_num_frames = 100,
)
OptimizationParams = dict(
    maxtime = 100,
    iterations = 1500,
    densify_until_iter = 1500,
    position_lr_max_steps = 1500,
    deformation_lr_max_steps = 1500,

    radegs_regularization_from_iter = 90_000
)