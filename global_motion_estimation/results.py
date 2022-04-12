from utils import get_video_frames, draw_motion_field, PSNR
import motion as motion
from json import dump
import numpy as np
import shutil
import cv2
import os

FRAME_DISTANCE = 3

video_list = ["numeri"]

if __name__ == "__main__":
    """Once set the video path and save path creates a lot of data for your report. Namely, it saves frames, compensated frames, frame differences and estimations of global motion.
    """
    for video in video_list:

        video_path = os.path.join("resources", "videos", str(video+".mp4"))
        save_path = os.path.join("results", video, "")


        psnr_dict = {}
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, "frames", ""))
        os.mkdir(os.path.join(save_path, "compensated", ""))
        os.mkdir(os.path.join(save_path, "curr_prev_diff", ""))
        os.mkdir(os.path.join(save_path, "model_motion_field", ""))
        os.mkdir(os.path.join(save_path, "curr_comp_diff", ""))

        frames = get_video_frames(video_path)
        print("frame shape: {}".format(frames[0].shape))
        for idx in range(FRAME_DISTANCE, len(frames)):

            j = (idx + 1) / len(frames)
            print("[%-20s] %d/%d frames" % ("=" * int(20 * j), idx, len(frames)))

            # get previous, current and compensated
            previous = frames[idx - FRAME_DISTANCE]
            current = frames[idx]

            params = motion.global_motion_estimation(previous, current)

            model_motion_field = motion.get_motion_field_affine(
                (int(previous.shape[0] / motion.BBME_BLOCK_SIZE), int(previous.shape[1] / motion.BBME_BLOCK_SIZE), 2), parameters=params
            )


            shape = (previous.shape[0]//motion.BBME_BLOCK_SIZE, previous.shape[1]//motion.BBME_BLOCK_SIZE)
            # compensate camera motion on previous frame
            compensated = motion.compensate_frame(previous, model_motion_field)


            idx_name = str(idx)
            # save frames
            cv2.imwrite(
                os.path.join(save_path, "frames", "")
                + str(idx-5)
                + ".png",
                previous,
            )

            # save compensated
            cv2.imwrite(
                os.path.join(save_path, "compensated", "")
                + str(idx-5)
                + ".png",
                compensated,
            )

            # save differences
            diff_curr_prev = (
                np.absolute(current.astype("int") - previous.astype("int"))
            ).astype("uint8")
            diff_curr_comp = (
                np.absolute(current.astype("int") - compensated.astype("int"))
            ).astype("uint8")
            cv2.imwrite(
                os.path.join(save_path, "curr_prev_diff", "")
                + str(idx)
                + ".png",
                diff_curr_prev,
            )
            cv2.imwrite(
                os.path.join(save_path, "curr_comp_diff", "")
                + str(idx)
                + ".png",
                diff_curr_comp,
            )

            # save motion model motion field
            draw = draw_motion_field(previous, model_motion_field)
            cv2.imwrite(
                os.path.join(save_path, "model_motion_field", "")
                + str(idx)
                + ".png",
                draw,
            )

            ## compute PSNR
            psnr = PSNR(current, compensated)
            psnr_dict[idx_name] = str(psnr)
            with open(save_path + "psnr_records.json", "w") as outfile:
                dump(psnr_dict, outfile)