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
    # parameters of the affine model
    parameters = motion.global_motion_estimation(previous, current)
    shape = (previous.shape[0]//motion.BBME_BLOCK_SIZE, previous.shape[1]//motion.BBME_BLOCK_SIZE)
    # motion field due to global motion
    motion_field = motion.get_motion_field_affine(shape, parameters)
    # compensate camera motion on previous frame
    compensated = compensate_frame(prev, motion_field)
    return compensated



if __name__ == "__main__":
    import utils
    import cv2
    import os
    from json import dump


    save_path = os.path.join(os.path.abspath('.'), 'results', 'pan240_mse', '')
    if not os.path.isdir(save_path):
        try:
            os.mkdir(save_path)
        except:
            raise Exception("The save path specified is not a folder")
    if not os.path.isdir(save_path + "frames"):
        os.mkdir(save_path + "frames")
        os.mkdir(save_path + "diff_curr_comp")
        os.mkdir(save_path + "diff_curr_prev")

    video_path = os.path.join(os.path.abspath('.'), 'resources', 'videos', 'pan240.mp4')
    # frames = utils.get_video_frames("./videos/Faster-Animation480.m4v")
    frames = utils.get_video_frames(video_path)
    psnr_dict = {}
    
    for i in range(3, len(frames)):
        prev = frames[i-3]
        curr = frames[i]
        
        compensated = motion_compensation(prev, curr)

        ## differences
        diff_curr_prev = (
            np.absolute(curr.astype("int") - prev.astype("int"))
        ).astype("uint8")
        diff_curr_comp = (
            np.absolute(curr.astype("int") - compensated.astype("int"))
        ).astype("uint8")

        idx_name = str(i - 3) + "-" + str(i)
        ## save results
        cv2.imwrite(save_path+"/frames/"+str(i-3)+"-frame.png", frames[i-3])
        cv2.imwrite(save_path+"/diff_curr_comp/"+str(idx_name)+".png", diff_curr_comp)
        cv2.imwrite(save_path+"/diff_curr_prev/"+str(idx_name)+".png", diff_curr_prev)
    
        ## compute PSNR
        psnr = utils.PSNR(curr, compensated)
        psnr_dict[idx_name] = str(psnr)
        with open(save_path + "psnr_records.json", "w") as outfile:
            dump(psnr_dict, outfile)