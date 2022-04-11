import numpy as np
import motion


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
        
        compensated = motion.motion_compensation(prev, curr)

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