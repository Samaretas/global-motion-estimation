import numpy as np
import motion

X = 1
Y = 0

def compensate_frame(frame, motion_field):
    """Compute the compensated frame given the starting frame and the motion field. Assumes the motion field was computed on squared blocks.

    Args:
        frame (np.ndarray): frame to be compensated.
        motion_field (np.ndarray): motion field.
    
    Returns:
        np.ndarray: the compensated frame.
    """
    compensated = np.copy(frame)
    # compute size of blocks
    bs = frame.shape[0]//motion_field.shape[0]
    for i in range(motion_field.shape[0]):
        for j in range(motion_field.shape[1]):
            a = i*bs
            d = motion_field[i,j]
            for _ in range(bs):
                b = j*bs
                for __ in range(bs):
                    try:
                        newa = a-d[X]
                        newb = b-d[Y]
                        assert newa > -1
                        assert newb > -1
                        compensated[a,b] = frame[newa,newb]
                    except:
                        pass
                    b += 1
                a += 1
    return compensated


def motion_compensation(previous, current):
    """Computes the compensated frame given the previous and the current frames.

    Args:
        previous (np.ndarray): previous frame.
        current (np.ndarray): current frame.

    Returns:
        np.ndarray: previous frame compensated w.r.t. the global motion estimated within the two frames.
    """
    parameters = motion.global_motion_estimation(previous, current)
    shape = (previous.shape[0]//motion.BBME_BLOCK_SIZE, previous.shape[1]//motion.BBME_BLOCK_SIZE)
    motion_field = motion.get_motion_field_affine(shape, parameters)
    compensated = compensate_frame(prev, motion_field)
    return compensated



if __name__ == "__main__":
    import utils
    import cv2

    # frames = utils.get_video_frames("./videos/Faster-Animation480.m4v")
    frames = utils.get_video_frames("./videos/pan240.mp4")
    prev = frames[30]
    curr = frames[33]
    
    compensated = motion_compensation(prev, curr)

    ## differences
    diff_curr_prev = (
        np.absolute(curr.astype("int") - prev.astype("int"))
    ).astype("uint8")
    diff_curr_comp = (
        np.absolute(curr.astype("int") - compensated.astype("int"))
    ).astype("uint8")

    ## draw global motion
    # prev = utils.draw_motion_field(prev, motion_field)

    ## save results
    cv2.imwrite("./resources/Previous.png", prev)
    cv2.imwrite("./resources/Current.png", curr[:,7:])
    cv2.imwrite("./resources/Compensated.png", compensated[:,7:])
    cv2.imwrite("./resources/diff_cur_prev.png", diff_curr_prev)
    cv2.imwrite("./resources/diff_cur_comp.png", diff_curr_comp)

    ## compute PSNR
    # print("PSNR {}".format(utils.PSNR(curr, compensated)))
    print("PSNR {}".format(utils.PSNR(curr[:,7:], compensated[:,7:])))