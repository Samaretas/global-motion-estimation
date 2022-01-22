from utils import get_video_frames
from motion import global_motion_estimation, first_estimation, compute_compensated_frame
import cv2
import numpy as np

frames = get_video_frames("./hall_objects_qcif.y4m")

for idx in range(30, len(frames)):
    previous = frames[idx-1]
    current = frames[idx]

    # global_motion_estimation(previous, current)
    parameters = first_estimation(previous, current)    # to test first estimation of the parameters
    compensated_frame = compute_compensated_frame(previous, parameters)
    # watch out for the difference
    difference = (np.absolute(compensated_frame.astype('int')-current.astype('int'))).astype('uint8')
    cv2.imshow("current", current)
    cv2.imshow("compensated", compensated_frame)
    cv2.imshow("difference", difference)
    cv2.waitKey(2)

print("Motion compensation tested")



