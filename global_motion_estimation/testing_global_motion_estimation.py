from utils import get_video_frames
import global_motion_estimation.motion as motion
import time, cv2
import numpy as np

if __name__ == "__main__":
    frames = get_video_frames("./hall_objects_qcif.y4m")
    # frames = get_video_frames("./foreman.mp4")

    for idx in range(30, len(frames)):
        previous = frames[idx-1]
        current = frames[idx]

        new_shape = (current.shape[0]*2,current.shape[0]*2) 

        # previous = cv2.resize(previous, new_shape)
        # current = cv2.resize(current, new_shape)

        start = time.time()
        compensated = motion.global_motion_estimation(previous, current)
        end = time.time()
        cv2.imshow("0", previous)
        cv2.imshow("1", current)
        cv2.imshow("2", compensated)
        cv2.waitKey(0)
        print(f"GME duration {int(end-start)}s")
        displacement = (np.absolute(current.astype('int')-previous.astype('int'))).astype('uint8')
        actual_motion = np.absolute(compensated.astype('int')-current.astype('int'))
        actual_motion = actual_motion.astype('uint8')
        cv2.imshow("displacement", displacement)
        cv2.imshow("actual motion", actual_motion)
        cv2.waitKey(0)
        break

    print("hessian GD tested")