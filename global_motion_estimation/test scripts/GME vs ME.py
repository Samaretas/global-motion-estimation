from cv2 import imshow
from utils import get_video_frames, PSNR, draw_motion_field
from bbme import get_motion_field
import motion as motion
from json import dump
import numpy as np
import argparse
import os
import cv2


if __name__ == "__main__":
    """
        Use this script to see the difference in motion fields computed using normal motion estimation (BBME in this version) vs using global motion estimaion (GME compensating current frame).

        Need to pass as parameters:
            -vp, path to the video you want to analyze
            -sp, path to folder where to save all the results
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-vp", "--video_path", help="path to the video you want to analyze", type=str
    )
    parser.add_argument(
        "-sp",
        "--save_path",
        help="path to folder where to save all the results",
        type=str,
    )
    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        try:
            os.mkdir(args.save_path)
        except:
            raise Exception("The save path specified is not a folder")
    if not os.path.isdir(args.save_path + "bbme"):
        os.mkdir(args.save_path + "bbme")
        os.mkdir(args.save_path + "gme")
        os.mkdir(args.save_path + "diff_curr_comp")
        os.mkdir(args.save_path + "diff_curr_prev")
        os.mkdir(args.save_path + "comp_differences")
    frames = get_video_frames(args.video_path)
    psnr_dict = {}
    prev_comp = None
    print("frame shape: {}".format(frames[0].shape))
    for idx in range(2, len(frames)):

        j = (idx + 1) / len(frames)
        print("[%-20s] %d/%d frames" % ("=" * int(20 * j), idx, len(frames)))

        # get previous, current and compensated
        previous = frames[idx - 2]
        current = frames[idx]
        compensated = motion.compensate_previous_frame(previous, current)

        # in case of need, filter here
        # compensated = cv2.medianBlur(compensated, ksize=3)

        # compute motion field both for previous and compensated
        motion_field_previous = get_motion_field(
            previous, current, block_size=10, searching_procedure=3
        )
        motion_field_compensated = get_motion_field(
            compensated, current, block_size=10, searching_procedure=3
        )

        ## differences
        diff_curr_prev = (
            np.absolute(current.astype("int") - previous.astype("int"))
        ).astype("uint8")
        diff_curr_comp = (
            np.absolute(current.astype("int") - compensated.astype("int"))
        ).astype("uint8")

        idx_name = str(idx - 1) + "-" + str(idx)
        draw = draw_motion_field(previous, motion_field_previous)
        cv2.imwrite(args.save_path + "bbme\\" + idx_name + ".png", draw)
        draw = draw_motion_field(previous, motion_field_compensated)
        cv2.imwrite(args.save_path + "gme\\" + idx_name + ".png", draw)
        cv2.imwrite(
            args.save_path + "diff_curr_comp\\" + idx_name + ".png", diff_curr_comp
        )
        cv2.imwrite(
            args.save_path + "diff_curr_prev\\" + idx_name + ".png", diff_curr_prev
        )

        # motion between compensated
        if prev_comp is None:
            prev_comp = compensated
        else:
            motion_field_compensated_compensated = get_motion_field(
                prev_comp, compensated, block_size=10, searching_procedure=3
            )
            draw = draw_motion_field(prev_comp, motion_field_compensated_compensated)
            cv2.imwrite(args.save_path + "comp_differences\\" + idx_name + ".png", draw)
            prev_comp = compensated

        psnr = PSNR(current, compensated)
        psnr_dict[idx_name] = str(psnr)
        with open(args.save_path + "psnr_records.json", "w") as outfile:
            dump(psnr_dict, outfile)