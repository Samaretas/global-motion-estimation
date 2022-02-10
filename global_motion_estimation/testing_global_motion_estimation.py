from utils import get_video_frames, PSNR
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
        cv2.imwrite("./results/prev.png", previous)
        cv2.imshow("prev", previous)
        cv2.imwrite("./results/curr.png", current)
        cv2.imshow("curr", current)
        cv2.imwrite("./results/compensated.png", compensated)
        cv2.imshow("comp", compensated)
        cv2.waitKey(0)
        frame_difference = (
            np.absolute(current.astype("int") - previous.astype("int"))
        ).astype("uint8")
        frame_diff_with_compensated = (
            np.absolute(compensated.astype("int") - current.astype("int"))
        ).astype("uint8")
        cv2.imwrite("./results/frame_difference.png", frame_difference)
        cv2.imshow("frame_difference", frame_difference)
        cv2.imwrite("./results/frame diff with compensated.png", frame_diff_with_compensated)
        cv2.imshow("frame diff with compensated", frame_diff_with_compensated)
        cv2.waitKey(0)
        psnr = PSNR(current, compensated)
        print(f"PSNR value:{psnr}")
        break
