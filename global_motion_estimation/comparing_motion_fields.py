from cv2 import imshow
from utils import get_video_frames, PSNR, draw_motion_field
from bbme import get_motion_fied
import motion as motion
import os
import cv2


video_path = ".\\videos\\pan240.mp4"
save_path = ".\\results\\pan_robust_affine\\"

if __name__ == "__main__":
    """Computes the motion field using BMME and GME, then shows the result to make a visual comparison between the two methods."""

    if not os.path.isdir(save_path):
        try:
            os.mkdir(save_path)
        except:
            raise Exception("The save path specified is not a folder")
    frames = get_video_frames(video_path)
    prev_comp = None
    print("frame shape: {}".format(frames[0].shape))
    for idx in range(6, len(frames)):

        j = (idx + 1) / len(frames)
        print("[%-20s] %d/%d frames" % ("=" * int(20 * j), idx, len(frames)))

        # get previous, current and compensated
        previous = frames[idx - 5]
        current = frames[idx]

        params = motion.global_motion_estimation(previous, current)

        # compute motion field ground truth and global
        estimated_motion_field = get_motion_fied(
            previous, current, block_size=motion.BMME_BLOCK_SIZE, searching_procedure=3
        )
        
        model_motion_field = motion.motion_field_affine(estimated_motion_field.shape, parameters=params)

        compensated_motion_field = estimated_motion_field-model_motion_field

        idx_name = str(idx - 1) + "-" + str(idx)
        draw_gt = draw_motion_field(previous, estimated_motion_field)
        draw_mm = draw_motion_field(previous, model_motion_field)
        draw_comp = draw_motion_field(previous, compensated_motion_field)

        # cv2.imshow(idx_name+" ground truth", draw_gt)
        # cv2.imshow(idx_name+" motion model", draw_mm)
        cv2.imwrite(save_path+idx_name+" GT.png", draw_gt)
        cv2.imwrite(save_path+idx_name+" MM.png", draw_mm)
        cv2.imwrite(save_path+idx_name+" Zcomp.png", draw_comp)