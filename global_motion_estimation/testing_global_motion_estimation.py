from utils import get_video_frames
import motion as motion
import cv2
import numpy as np

if __name__ == "__main__":
    # frames = get_video_frames("./hall_objects_qcif.y4m")
    frames = get_video_frames("./global_motion_estimation/translation.mp4")

    for idx in range(23, len(frames)):
        previous = frames[idx - 1]
        current = frames[idx]

        new_shape = (int(current.shape[0] / 2), int(current.shape[0] / 2))

        previous = cv2.resize(previous, new_shape)
        current = cv2.resize(current, new_shape)

        compensated = motion.global_motion_estimation(previous, current)
        cv2.imshow("prev", previous)
        cv2.imshow("curr", current)
        cv2.imshow("comp", compensated)
        cv2.waitKey(0)
        displacement = (
            np.absolute(current.astype("int") - previous.astype("int"))
        ).astype("uint8")
        actual_motion = (
            np.absolute(compensated.astype("int") - current.astype("int"))
        ).astype("uint8")
        cv2.imshow("displacement", displacement)
        cv2.imshow("actual motion", actual_motion)
        cv2.waitKey(0)
        break
