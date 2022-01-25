from utils import get_video_frames
import global_motion_estimation.motion as motion
import time, cv2
import numpy as np

frames = get_video_frames("./hall_objects_qcif.y4m")

for idx in range(30, len(frames)):
    previous = frames[idx-1]
    current = frames[idx]

    # cv2.imshow("cur", current)
    # cv2.waitKey(1)

    # global_motion_estimation(previous, current)
    parameters = motion.first_estimation(previous, current)    # to test first estimation of the parameters

    compensated_pre_update = motion.compute_compensated_frame(previous, parameters)
    print(f"original parameters\n{parameters}")
    start = time.time()
    updated = motion.handmade_gradient_descent(parameters, previous, current)
    end = time.time()
    print(f"updated parameters\n{updated}, duration {int(end-start)}s")
    compensated_post_update = motion.compute_compensated_frame(previous, updated)
    diff1 = (np.absolute(compensated_pre_update.astype('int')-current.astype('int'))).astype('uint8')
    diff2 = (np.absolute(compensated_post_update.astype('int')-current.astype('int'))).astype('uint8')
    cv2.imshow("pre update", diff1)
    cv2.imshow("post update", diff2)
    cv2.imshow("current", current)
    cv2.waitKey(0)
    break

print("hessian GD tested")