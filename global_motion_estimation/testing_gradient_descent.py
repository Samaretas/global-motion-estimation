from utils import get_video_frames
import motion as motion
import cv2
import numpy as np

# if __name__ == "__main__":
if True:
    # frames = get_video_frames("./hall_objects_qcif.y4m")
    frames = get_video_frames("./global_motion_estimation/translation.mp4") # 33 frames

    for idx in range(23, len(frames)):
        previous = frames[idx-1]
        current = frames[idx]

        new_shape = (int(previous.shape[0]/2), int(previous.shape[1]/2))    
        previous = cv2.resize(previous, new_shape)
        current = cv2.resize(current, new_shape)

        cv2.imshow("prev", previous)

        # global_motion_estimation(previous, current)
        parameters = motion.first_estimation(previous, current)    # to test first estimation of the parameters
        compensated_pre_update = motion.compute_compensated_frame(previous, parameters)

        print(f"original parameters\n{parameters}")
        updated = motion.handmade_gradient_descent(parameters, previous, current)
        # updated = motion.handmade_gradient_descent_mp(parameters, previous, current)
        print(f"updated parameters\n{updated}")

        # compensated_post_update = motion.compute_compensated_frame_with_log(previous, updated)
        compensated_post_update = motion.compute_compensated_frame_complete(previous, updated)
        diff1 = (np.absolute(compensated_pre_update.astype('int')-current.astype('int'))).astype('uint8')
        diff2 = (np.absolute(compensated_post_update.astype('int')-current.astype('int'))).astype('uint8')
        np.savetxt("compensated_post_update.txt", compensated_post_update, fmt="%-3d")
        np.savetxt("diff2.txt", diff2, fmt="%-3d")
        cv2.imshow("current", current)
        cv2.imshow("predicted wout updataed params", compensated_pre_update)
        cv2.imshow("predicted w updataed params", compensated_post_update)
        cv2.imshow("pre update", diff1)
        cv2.imshow("post update", diff2)
        cv2.waitKey(0)
        break
