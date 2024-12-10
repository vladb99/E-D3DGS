_base_ = './default.py'

ModelParams = dict(
    disable_filter3D=False,
)

ModelHiddenParams = dict(
    total_num_frames = 100,
)
OptimizationParams = dict(
    maxtime = 100,
    iterations = 80_000,
    densify_until_iter = 80_000,
    position_lr_max_steps = 80_000,
    deformation_lr_max_steps = 80_000,

    radegs_regularization_from_iter = 15_000
)