from scene import Scene


def get_time_steps(scene: Scene) -> [float]:
    time_steps = []
    # We use the video cameras to just get the values of the timesteps, because there is one video camera per time frame.
    # in getTrainCameras(), we have multiple cameras per timeframe
    for cam in scene.getVideoCameras():
        time_steps.append(cam.time)
    return time_steps