from utils import get_video_frames
from motion import first_estimation, compute_compensated_frame
import cv2
import numpy as np

# frames = get_video_frames("./hall_objects_qcif.y4m")
frames = get_video_frames("./translation.mp4")  # 33 frames

for idx in range(23, len(frames)):
    previous = frames[idx - 1]
    current = frames[idx]

    # global_motion_estimation(previous, current)
    parameters = first_estimation(
        previous, current
    )  # to test first estimation of the parameters
    compensated_frame = compute_compensated_frame(previous, parameters)
    # watch out for the difference
    difference = (
        np.absolute(compensated_frame.astype("int") - current.astype("int"))
    ).astype("uint8")
    cv2.imshow("current", current)
    cv2.imshow("compensated", compensated_frame)
    cv2.imshow("difference", difference)
    cv2.waitKey(0)
    break

print("Motion compensation tested")
