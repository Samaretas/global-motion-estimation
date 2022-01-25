import cv2
import numpy
from time import time
from utils import get_video_frames, get_pyramids
from global_motion_estimation.motion import (
    compute_compensated_frame,
    first_estimation,
    handmade_gradient_descent,
    handmade_gradient_descent_mp,
)

if __name__ == "__main__":
    start = time()
    frames = get_video_frames("hall_objects_qcif.y4m")
    print("{:.2f}s to get the video".format(time() - start))
    start = time()

    prev = frames[31]
    curr = frames[32]

    pyr1 = get_pyramids(prev)
    pyr1 = get_pyramids(curr)
    print("{:.2f}s to get the pyramids".format(time() - start))
    start = time()

    parameters = first_estimation(prev, curr)
    print("{:.2f}s to get the first estimate".format(time() - start))
    start = time()

    updated = handmade_gradient_descent_mp(parameters, prev, curr)
    print("{:.2f}s to gradient".format(time() - start))

    compensated = compute_compensated_frame(prev, parameters)

    displacement = (numpy.absolute(curr.astype("int") - prev.astype("int"))).astype(
        "uint8"
    )
    corrected = (numpy.absolute(compensated.astype("int") - prev.astype("int"))).astype(
        "uint8"
    )
    cv2.imshow("displacement", displacement)
    cv2.imshow("corrected", corrected)
    cv2.waitKey(0)