from utils import get_video_frames
from motion import global_motion_estimation
import cv2
import numpy as np

frames = get_video_frames("./hall_objects_qcif.y4m")

for idx in range(1, len(frames)):
    precedent = frames[idx-1]
    current = frames[idx]

    # cv2.imshow("cur", current_pyr[0])
    # cv2.waitKey(0)

    global_motion_estimation(precedent, current)