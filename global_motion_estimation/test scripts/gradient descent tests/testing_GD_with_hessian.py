from utils import get_video_frames
import global_motion_estimation.motion as motion
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
    compensated_frame = motion.compute_compensated_frame(previous, parameters=parameters)
    hessian = main(current, previous, 10, 10)

    break

print("hessian GD tested")