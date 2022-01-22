from utils import get_video_frames
import motion
import time, cv2
import numpy as np

if __name__ == "__main__":
    frames = get_video_frames("./hall_objects_qcif.y4m")
    # frames = get_video_frames("./foreman.mp4")

    for idx in range(30, len(frames)):
        previous = frames[idx-1]
        current = frames[idx]

        start = time.time()
        actual_motion = motion.global_motion_estimation(previous, current)
        end = time.time()
        cv2.imshow("current", current)
        cv2.imshow("current", previous)
        print(f"GME duration {int(end-start)}s")
        displacement = (np.absolute(current.astype('int')-previous.astype('int'))).astype('uint8')
        cv2.imshow("displacement", displacement)
        cv2.imshow("actual motion", actual_motion)
        cv2.waitKey(0)
        break

    print("hessian GD tested")