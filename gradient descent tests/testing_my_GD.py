from utils import get_video_frames
import motion
import cv2
import numpy as np
from hessian_gradient import main

frames = get_video_frames("./hall_objects_qcif.y4m")

for idx in range(30, len(frames)):
    previous = frames[idx-1]
    current = frames[idx]

    # cv2.imshow("cur", current)
    # cv2.waitKey(1)

    # global_motion_estimation(previous, current)
    parameters = motion.first_estimation(previous, current)    # to test first estimation of the parameters
    print(f"original parameters\n{parameters}")
    updated = motion.parameter_optimization(parameters, previous, current)
    print(f"updated parameters\n{updated}")
    break

print("hessian GD tested")